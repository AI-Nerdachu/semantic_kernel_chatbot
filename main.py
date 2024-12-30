import asyncio
import logging
from datetime import datetime
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore
from agents.custom_chat_completion_agent import CustomChatCompletionAgent
from agents.custom_retrieval_agent import CustomRetrievalAgent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from utils.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_COGNITIVE_SEARCH_ENDPOINT,
    AZURE_COGNITIVE_SEARCH_ADMIN_KEY,
    AZURE_COGNITIVE_SEARCH_INDEX_NAME,
    AZURE_OPENAI_DEPLOYMENT_NAME
)
from utils.logger import setup_logging

# Logging setup
logger = setup_logging(log_level="INFO", log_to_file=True)

async def initialize_kernel_and_agents():
    """
    Initialize the Semantic Kernel and custom agents with detailed logging.
    """
    try:
        logger.info("Initializing Semantic Kernel...")
        kernel = Kernel()

        # Configure Azure OpenAI Services
        logger.info("Configuring Azure OpenAI services...")
        chat_service = AzureChatCompletion(
            service_id="chat_service",
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        )
        embedding_service = AzureTextEmbedding(
            service_id="embedding_service",
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        )

        kernel.add_service(service=chat_service)
        kernel.add_service(service=embedding_service)
        logger.info("Azure OpenAI services configured successfully.")

        # Configure Azure Cognitive Search
        logger.info("Configuring Azure Cognitive Search...")
        acs_connector = AzureCognitiveSearchMemoryStore(
            vector_size=1536,
            search_endpoint=AZURE_COGNITIVE_SEARCH_ENDPOINT,
            admin_key=AZURE_COGNITIVE_SEARCH_ADMIN_KEY,
        )
        await acs_connector.create_collection(collection_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME)
        acs_connector.service_id = "memory_service"
        kernel.add_service(service=acs_connector)
        logger.info("Azure Cognitive Search configured successfully.")

        # Initialize agents
        logger.info("Initializing custom agents...")
        chat_agent = CustomChatCompletionAgent(
            kernel=kernel, instructions="Be a friendly and helpful assistant."
        )
        retrieval_agent = CustomRetrievalAgent(
            search_endpoint=AZURE_COGNITIVE_SEARCH_ENDPOINT,
            api_key=AZURE_COGNITIVE_SEARCH_ADMIN_KEY,
            index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME,
        )
        logger.info("Custom agents initialized successfully.")

        return kernel, chat_agent, retrieval_agent

    except Exception as e:
        logger.critical("Failed to initialize Semantic Kernel and agents.", exc_info=True)
        raise RuntimeError("Initialization of Kernel or agents failed.") from e

from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent

async def main():
    """
    Main function for chatbot interaction with detailed logging.
    """
    try:
        logger.info("Starting chatbot...")
        kernel, chat_agent, retrieval_agent = await initialize_kernel_and_agents()
        logger.info("Chatbot initialized successfully.")

        # Initialize chat history
        history = ChatHistory()

        logger.info("Chatbot ready. Type 'exit' to quit.")
        print("Chatbot ready. Type 'exit' to quit.")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() == "exit":
                    logger.info("User exited the chatbot.")
                    print("Goodbye!")
                    break

                if user_input:
                    # Add user input to the chat history
                    user_message = ChatMessageContent(role="user", content=user_input)
                    history.add_message(user_message)

                    logger.info("Processing chat interaction...")
                    async for response in chat_agent.invoke(history):
                        print(f"Assistant: {response.content}")
                        history.add_message(response)
                    logger.info("Chat interaction completed.")

            except Exception as e:
                logger.error("Unexpected error in main loop: %s", e, exc_info=True)
                print("An error occurred. Please try again later.")

    except Exception as e:
        logger.critical("Critical error during chatbot execution: %s", e, exc_info=True)
        print("Critical error occurred. Unable to start chatbot.")



if __name__ == "__main__":
    asyncio.run(main())