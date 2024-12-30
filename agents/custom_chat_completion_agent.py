from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
import logging
from datetime import datetime
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.exceptions import KernelServiceNotFoundError

logger = logging.getLogger(__name__)

class CustomChatCompletionAgent(ChatCompletionAgent):
    """
    Custom Chat Completion Agent with advanced logging, validation, and error handling.
    """

    async def invoke(self, history):
        """
        Invoke the chat completion service with detailed logging and error handling.

        Args:
            history: Either a ChatHistory instance or a list of ChatMessageContent instances.

        Yields:
            ChatMessageContent: The assistant's responses.
        """
        try:
            # Convert list to ChatHistory if necessary
            if isinstance(history, list):
                logger.debug("Converting history list to ChatHistory instance.")
                chat_history = ChatHistory(messages=history)
            else:
                chat_history = history

            logger.info("Invoking ChatCompletionAgent with %d message(s) in history.", len(chat_history.messages))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("History Content: %s", [msg.content for msg in chat_history.messages])

            start_time = datetime.now()

            # Use parent class's invoke method for service call
            async for message in super().invoke(chat_history):
                logger.info("Response received: %s", message.content)
                self._log_response_details(message)
                yield message

            logger.info("Processing completed in %.2f seconds.", (datetime.now() - start_time).total_seconds())

        except KernelServiceNotFoundError as e:
            logger.error("Kernel service not found: %s", e, exc_info=True)
            yield ChatMessageContent(
                role="system", content="Error: Required chat service is not available. Please try again later."
            )
        except Exception as e:
            logger.error("Unexpected error occurred during invocation: %s", e, exc_info=True)
            yield ChatMessageContent(role="system", content="Error: An unexpected issue occurred. Please try again.")

    def _log_response_details(self, message):
        """
        Log details of a response for debugging or analytics.

        Args:
            message: The ChatMessageContent to log.
        """
        logger.info(
            "Response Details - Role: %s, Content Length: %d, Timestamp: %s",
            message.role,
            len(message.content) if message.content else 0,
            datetime.now().isoformat(),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Full Response Content: %s", message.content)



    def _validate_history(self, history):
        """
        Validate the input chat history.

        Args:
            history: List of chat messages to validate.

        Returns:
            bool: True if valid, raises an exception otherwise.
        """
        if not isinstance(history, list):
            logger.error("Chat history must be a list, received type: %s", type(history))
            raise ValueError("Chat history must be a list.")

        if not all(isinstance(msg, ChatMessageContent) for msg in history):
            logger.error("Chat history contains invalid message types.")
            raise ValueError("All history items must be instances of ChatMessageContent.")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Chat history validation passed with %d message(s).", len(history))

        return True

    def _log_response_details(self, message):
        """
        Log details of a response for debugging or analytics.

        Args:
            message: The ChatMessageContent to log.
        """
        logger.info(
            "Response Details - Role: %s, Content Length: %d, Timestamp: %s",
            message.role,
            len(message.content) if message.content else 0,
            datetime.now().isoformat(),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Full Response Content: %s", message.content)

    async def invoke_with_validation(self, history):
        """
        Invoke the chat completion service with input validation.

        Args:
            history: List of chat messages.

        Yields:
            ChatMessageContent: The assistant's responses.
        """
        try:
            logger.info("Validating chat history before invoking the service...")
            self._validate_history(history)

            async for message in self.invoke(history):
                yield message

        except ValueError as e:
            logger.error("Validation error: %s", e, exc_info=True)
            yield ChatMessageContent(
                role="system", content="Error: Invalid input provided. Please provide a valid chat history."
            )
        except KernelServiceNotFoundError as e:
            logger.error("Kernel service not found during validation: %s", e, exc_info=True)
            yield ChatMessageContent(role="system", content="Error: Chat service is currently unavailable.")
        except Exception as e:
            logger.error("Unexpected error during invocation with validation: %s", e, exc_info=True)
            yield ChatMessageContent(role="system", content="Error: An unexpected issue occurred. Please try again.")
