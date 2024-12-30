from models.chat_response import ChatResponse, ErrorResponse

def test_chat_response():
    # Test successful response
    response = ChatResponse(
        assistant="Hello, how can I help?",
        timestamp="2024-12-24T15:00:00Z"
    )
    print(response.json())

    # Test error response
    error_response = ErrorResponse(
        error="An unexpected error occurred.",
        detail="Kernel not initialized."
    )
    print(error_response.json())

if __name__ == "__main__":
    test_chat_response()
