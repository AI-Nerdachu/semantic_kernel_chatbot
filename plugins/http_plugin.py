import requests
from typing import Optional
from semantic_kernel.functions.kernel_function_decorator import kernel_function

class HttpPlugin:
    """
    A plugin to perform HTTP GET and POST requests.
    """

    @kernel_function(
        name="get",
        description="Perform an HTTP GET request to the provided URL and return the response text."
    )
    def get(self, url: str, headers: Optional[dict] = None) -> str:
        """
        Perform an HTTP GET request.

        Args:
            url (str): The target URL.
            headers (Optional[dict]): Optional headers for the request.

        Returns:
            str: The response text or an error message.
        """
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"HTTP GET request failed: {str(e)}"

    @kernel_function(
        name="post",
        description="Perform an HTTP POST request to the provided URL with optional data, returning the response text."
    )
    def post(self, url: str, data: Optional[dict] = None, headers: Optional[dict] = None) -> str:
        """
        Perform an HTTP POST request.

        Args:
            url (str): The target URL.
            data (Optional[dict]): The payload to send in the POST request.
            headers (Optional[dict]): Optional headers for the request.

        Returns:
            str: The response text or an error message.
        """
        try:
            response = requests.post(url, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            return f"HTTP POST request failed: {str(e)}"
