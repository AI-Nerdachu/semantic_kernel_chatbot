import logging
from semantic_kernel.agents import Agent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.exceptions import KernelServiceNotFoundError

logger = logging.getLogger(__name__)

class CustomSummarizationAgent(Agent):
    """
    A custom agent for summarizing text or conversations using Semantic Kernel.
    """

    def __init__(self, kernel, service_id="summarization_service"):
        """
        Initialize the Summarization Agent.

        Args:
            kernel: The Semantic Kernel instance.
            service_id: The service ID for the summarization model.
        """
        super().__init__()
        self.kernel = kernel
        self.service_id = service_id
        logger.info("CustomSummarizationAgent initialized with service ID: %s", service_id)

    async def summarize_text(self, text: str) -> str:
        """
        Summarize a block of text.

        Args:
            text: The input text to summarize.

        Returns:
            str: The summarized text.
        """
        if not text.strip():
            logger.warning("Empty text received for summarization.")
            return "No content to summarize."

        try:
            logger.info("Summarizing text with service ID: %s", self.service_id)
            prompt = f"Summarize the following text:\n\n{text}"
            logger.debug("Summarization prompt: %s", prompt)
            result = await self._invoke_service(prompt)
            logger.info("Text summarization completed successfully.")
            return result
        except KernelServiceNotFoundError as e:
            logger.error("Summarization service not found: %s", e, exc_info=True)
            return "Summarization service is currently unavailable."
        except Exception as e:
            logger.error("Error during text summarization: %s", e, exc_info=True)
            return "An error occurred while summarizing the text."

    async def summarize_conversation(self, messages: list[ChatMessageContent]) -> str:
        """
        Summarize a conversation from chat history.

        Args:
            messages: A list of ChatMessageContent objects representing the conversation.

        Returns:
            str: The summarized conversation.
        """
        if not messages:
            logger.warning("Empty conversation history received for summarization.")
            return "No conversation history to summarize."

        try:
            logger.info("Summarizing conversation with service ID: %s", self.service_id)
            conversation_text = "\n".join(
                f"{msg.role.value}: {msg.content.strip()}" for msg in messages if msg.content.strip()
            )
            prompt = f"Summarize the following conversation:\n\n{conversation_text}"
            logger.debug("Conversation summarization prompt: %s", prompt)
            result = await self._invoke_service(prompt)
            logger.info("Conversation summarization completed successfully.")
            return result
        except KernelServiceNotFoundError as e:
            logger.error("Summarization service not found: %s", e, exc_info=True)
            return "Summarization service is currently unavailable."
        except Exception as e:
            logger.error("Error during conversation summarization: %s", e, exc_info=True)
            return "An error occurred while summarizing the conversation."

    async def _invoke_service(self, prompt: str) -> str:
        """
        Invoke the summarization service with a given prompt.

        Args:
            prompt: The prompt for the summarization service.

        Returns:
            str: The result from the summarization service.
        """
        try:
            service = self.kernel.get_service(service_id=self.service_id)
            if not service:
                raise KernelServiceNotFoundError(f"Service not found: {self.service_id}")

            logger.info("Invoking summarization service...")
            response = await service.complete(prompt, timeout=10)  # Add a timeout for robustness
            logger.debug("Summarization service response: %s", response.text)
            return response.text.strip()
        except KernelServiceNotFoundError:
            raise  # Re-raise to handle it higher up
        except Exception as e:
            logger.error("Error invoking summarization service: %s", e, exc_info=True)
            raise RuntimeError("Failed to invoke summarization service.") from e
