# Copyright 2024 FBL-A, Inc.
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

import argparse
import json
import logging
import os
import time

import ray
import torch
from fbl.util.hydra import HydraConfig
from fbl.util.logging import setup_logging
from fbl.util.ray import RayClassWithInitArgs, get_head_node_ip, init_ray

from verl.data_access.data_access import VerlDataAccess
from verl.proto.data_pb2 import DataProto
from verl.single_controller.reward_manager import RewardManager
from verl.utils.checkpoint import copy_to_local
from verl.utils.config_util import get_worker_config
from verl.utils.data_util import dataproto_to_list, list_to_dataproto
from verl.utils.ray_util import get_worker_group_size
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to a file with pre-generated rollouts (JSONL format). If provided, skips generation.",
    )
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the scored results (JSONL format)")
    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="final_answer",
        choices=["chain_of_thought", "final_answer"],
        help="Scoring mode: 'chain_of_thought' or 'final_answer'",
    )
    parser.add_argument("--rollout_batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--scoring_batch_size", type=int, default=8, help="Batch size for scoring")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    config = HydraConfig(args.config_path)

    # Initialize Ray
    head_node_ip = get_head_node_ip()
    init_ray(config.ray_init.address, head_node_ip, config.ray_init.resources)
    logger.info(f"Ray initialized at {ray.get_dashboard_url()}")

    # Get worker configurations
    actor_config, reward_config = get_worker_config(
        config,
        ["actor_rollout_ref", "critic"],
        ["actor.strategy", "strategy"],
    )

    # === Initialize Reward Manager and Worker ===
    logger.info("Initializing Reward Manager and Worker...")
    reward_manager = RewardManager(config=config.critic)
    reward_worker_size = get_worker_group_size(reward_config, config.trainer.get("reward_model_pool_id"))
    reward_ray_cls = RayClassWithInitArgs(cls=ray.remote(RewardModelWorker), config=reward_config, role="critic")
    reward_model_wg = reward_ray_cls.create_worker_group(world_size=reward_worker_size)
    reward_model_wg.init_model()
    logger.info("Reward Manager and Worker initialized.")

    rollouts = []

    if args.input_file:
        # === Offline Scoring Mode ===
        logger.info(f"Loading pre-generated rollouts from {args.input_file}")
        with open(args.input_file, "r") as f:
            for line in f:
                rollouts.append(json.loads(line))
        prompts = [r["prompt"] for r in rollouts]
        generations = [r["generation"] for r in rollouts]

    else:
        # === Online Rollout and Scoring Mode ===
        logger.info("Initializing Policy Model for generation...")
        actor_worker_size = get_worker_group_size(actor_config, config.trainer.get("actor_rollout_ref_pool_id"))
        actor_ray_cls = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker),
            config=actor_config,
            role="rollout",
        )
        actor_wg = actor_ray_cls.create_worker_group(world_size=actor_worker_size)

        # Load checkpoint for policy model
        actor_path = config.actor_rollout_ref.model.path
        if not os.path.exists(actor_path):
            logger.info(f"Copying actor checkpoint from {actor_path} to local.")
            local_path = copy_to_local(actor_path)
            config.actor_rollout_ref.model.path = local_path
            actor_wg.load_checkpoint(local_path)
        else:
            actor_wg.load_checkpoint(actor_path)
        logger.info("Policy Model initialized and checkpoint loaded.")

        # Load dataset
        logger.info(f"Loading dataset from {config.data.val_files}")
        data_access = VerlDataAccess(config.data)
        dataset = data_access.get_val_dataloader()
        
        prompts = []
        generations = []
        
        logger.info("Generating rollouts...")
        for i, batch in enumerate(dataset):
            if i * args.rollout_batch_size > 10: # for testing
                break
            logger.info(f"Processing batch {i}")
            batch_prompts = dataproto_to_list(batch)
            prompts.extend(batch_prompts)
            
            output_batch = actor_wg.generate_sequences(batch)
            batch_generations = [
                output["response"] for output in dataproto_to_list(output_batch)
            ]
            generations.extend(batch_generations)
            rollouts.extend(
                [
                    {
                        "prompt": p,
                        "generation": g,
                        "model_source": config.actor_rollout_ref.model.path,
                        "data_source": config.data.val_files,
                    }
                    for p, g in zip(batch_prompts, batch_generations)
                ]
            )
        logger.info("Rollout generation complete.")


    # === Scoring ===
    logger.info("Starting scoring...")
    scores = []
    for i in range(0, len(generations), args.scoring_batch_size):
        logger.info(f"Scoring batch {i // args.scoring_batch_size}")
        batch_prompts = prompts[i:i + args.scoring_batch_size]
        batch_generations = generations[i:i + args.scoring_batch_size]

        if args.scoring_mode == "final_answer":
            # Use reward manager to parse and score
            parsed_answers = reward_manager.parse(batch_generations)
            
            data_for_scoring = list_to_dataproto(
                [
                    {"prompt": p, "response": pa}
                    for p, pa in zip(batch_prompts, parsed_answers)
                ]
            )
             g in zip(batch_prompts, batch_generations)
                ]
            )

        # Compute rewards using the reward model worker group
        reward_output = reward_model_wg.compute_reward(data_for_scoring)
        batch_scores = [d["reward"] for d in dataproto_to_list(reward_output)]
        scores.extend(batch_scores)

    logger.info("Scoring complete.")

    # === Save Results ===
    logger.info(f"Saving results to {args.output_file}")
    with open(args.output_file, "w") as f:
        for i, rollout in enumerate(rollouts):
            rollout["score"] = scores[i]
            rollout["scoring_mode"] = args.scoring_mode
            f.write(json.dumps(rollout) + "\n")

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
