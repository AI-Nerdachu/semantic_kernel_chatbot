import asyncio
import logging

from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent

from server import initialize_kernel_and_agents

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_agent():
    """
    Test the CustomChatCompletionAgent in isolation to debug issues.
    """
    # Initialize kernel and agents
    logger.info("Initializing Semantic Kernel and agents for testing...")
    kernel, chat_agent, _ = await initialize_kernel_and_agents()
    logger.info("Kernel and agents initialized successfully.")

    # Prepare chat history for testing
    history = ChatHistory(
        messages=[
            ChatMessageContent(
                role="system", content="You are an intent detection assistant."
            ),
            ChatMessageContent(role="user", content="Hello, how are you?"),
        ]
    )

    logger.info(
        "Prepared chat history for testing: %s",
        [msg.content for msg in history.messages],
    )

    # Test the agent
    try:
        logger.info("Invoking CustomChatCompletionAgent with test history...")
        async for message in chat_agent.invoke(history):
            logger.info("Response from agent: %s", message.content)
    except Exception as e:
        logger.error("Error during agent testing: %s", e, exc_info=True)


# Run the test
if __name__ == "__main__":
    asyncio.run(test_agent())
