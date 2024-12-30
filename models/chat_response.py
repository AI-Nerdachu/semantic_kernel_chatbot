from pydantic import BaseModel
from typing import Optional


class ChatResponse(BaseModel):
    """
    Model for successful chatbot responses.
    """
    assistant: str
    timestamp: Optional[str] = None
    status: str = "success"


class ErrorResponse(BaseModel):
    """
    Model for error responses.
    """
    error: str
    detail: Optional[str] = None
    status: str = "error"
