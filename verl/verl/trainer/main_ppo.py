# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import os
import ray
import hydra


def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        tokenizer = hf_tokenizer(local_path)

        # ---- START: Custom code for Qwen model output format ----
        # Define the system prompt with instructions for the desired output format.
        from verl.ours.prompt import SYSTEM_PROMPT
        system_prompt_instruction = SYSTEM_PROMPT

        # Create a Jinja2 chat template for Qwen that handles system prompts intelligently.
        # If the dataset provides a system prompt, use it. Otherwise, use our default instruction.
        qwen_custom_template = (
            "{% if messages[0]['role'] == 'system' %}"
                "{{ '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
            "{% else %}"
                f"{{{{ '<|im_start|>system\\n{system_prompt_instruction}<|im_end|>\\n' }}}}"
            "{% endif %}"
            "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                    "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

        tokenizer.chat_template = qwen_custom_template
        print("--- Applied custom Qwen chat template to enforce <think>/<answer> format. ---")
        # ---- END: Custom code ----

        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
        }

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        reward_manager_cls = None
        
        # New logic to handle single or multi-model names (e.g., "dsr1-internlm2")
        known_llm_judges = {'dsr1', 'internlm2','skywork','rmrb2',"skyworkqwen", "rule"}
        model_names = reward_manager_name.split('-')

        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        elif any(name in known_llm_judges for name in model_names):
            from verl.workers.reward_manager import LLMasJudgeRewardManager
            reward_manager_cls = LLMasJudgeRewardManager
        else:
            raise NotImplementedError(f"Unsupported reward_manager_name: {reward_manager_name}")
        
        if reward_manager_cls == LLMasJudgeRewardManager:
            compute_score = reward_manager_name + "_" + config.reward_model.get("reward_manager_type","")
        else:
            compute_score = get_custom_reward_fn(config)

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

        # Note that we always use function-based RM for validation
        val_reward_manager_name = config.reward_model.get("val_reward_manager", "dsr1")
        val_reward_manager_cls = None

        #already get known_llm_judges
        val_model_names = val_reward_manager_name.split('-')
        val_known_llm_judges = {'dsr1', 'internlm2','skywork','rmrb2',"skyworkqwen", "rule"} # Use the same set of known judges
        if val_reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            val_reward_manager_cls = NaiveRewardManager
        elif val_reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            val_reward_manager_cls = PrimeRewardManager
        elif any(name in val_known_llm_judges for name in val_model_names):
            from verl.workers.reward_manager import LLMasJudgeRewardManager
            val_reward_manager_cls = LLMasJudgeRewardManager
        else:
            raise NotImplementedError(f"Unsupported val_reward_manager_name: {val_reward_manager_name}")
        
        if val_reward_manager_cls == LLMasJudgeRewardManager:
            val_compute_score = val_reward_manager_name + "_" + config.reward_model.get("val_reward_manager_type","")
        else:
            val_compute_score = get_custom_reward_fn(config) 
        val_reward_fn = val_reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=val_compute_score)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                processor=processor,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()
