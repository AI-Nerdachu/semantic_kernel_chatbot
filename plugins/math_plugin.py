from semantic_kernel.functions.kernel_function_decorator import kernel_function

class MathPlugin:
    @kernel_function(name="add", description="Add two numbers.")
    def add(self, x: float, y: float) -> float:
        return x + y

    @kernel_function(name="subtract", description="Subtract two numbers.")
    def subtract(self, x: float, y: float) -> float:
        return x - y
