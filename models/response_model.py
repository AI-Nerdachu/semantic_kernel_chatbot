from pydantic import BaseModel
from typing import Optional

class ChatResponse(BaseModel):
    """
    Response model for the chatbot's `/chat` endpoint.
    """
    assistant: str
    timestamp: Optional[str] = None
    status: str = "success"  # Default value

class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    """
    error: str
    detail: Optional[str] = None
    status: str = "error"
