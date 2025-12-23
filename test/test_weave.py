import weave

# Initialize Weave Tracing
weave.init('intro-example')

# Decorate your function
@weave.op
def my_function(name: str):
    return f"Hello, {name}!"

# Call your function -- Weave will automatically track inputs and outputs
print(my_function("World"))