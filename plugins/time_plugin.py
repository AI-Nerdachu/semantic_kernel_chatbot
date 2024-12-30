import datetime
from semantic_kernel.functions.kernel_function_decorator import kernel_function

class TimePlugin:
    @kernel_function(name="current_time", description="Get current time.")
    def current_time(self) -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")

    @kernel_function(name="current_date", description="Get current date.")
    def current_date(self) -> str:
        return datetime.date.today().strftime("%Y-%m-%d")
