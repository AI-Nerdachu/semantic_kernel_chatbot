import json
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from utils.logger import setup_logging

logger = setup_logging(log_level="DEBUG", log_to_file=True)

class IntentDetectionPlugin:
    """A plugin to detect user intent dynamically using a CustomChatCompletionAgent."""

    def __init__(self, agent):
        """
        Initialize the IntentDetectionPlugin.

        Args:
            agent: An instance of CustomChatCompletionAgent.
        """
        self.agent = agent

    @kernel_function(
        description="Detect the user's intent based on the input message using an LLM.",
        name="DetectIntent",
    )
    async def detect_intent(self, kernel, input: str) -> str:
        logger.debug("Step 1: Starting 'detect_intent' function in IntentDetectionPlugin.")

        # Refined prompt
        logger.debug("Step 2: Preparing chat history for intent detection.")
        history = [
            ChatMessageContent(
                role="system",
                content="""
                You are a highly capable intent detection assistant. Your primary responsibility is to classify user inputs into specific intent categories and extract relevant details using the guidelines and plugins listed below.

                ### Available Plugins:
                1. **HTTP Plugin**:
                - Purpose: Perform general HTTP requests to fetch data from external APIs.
                - Examples: Retrieve data from a web service or an API.
                2. **WeatherAPI Plugin**:
                - Purpose: Provide current weather information or forecasts based on user-specified locations.
                - Examples: "What's the weather in Paris?" or "Will it rain tomorrow in New York?"
                3. **Math Plugin**:
                - Purpose: Solve equations, perform arithmetic, or handle complex mathematical operations.
                - Examples: "Calculate 25 + 75" or "Solve x^2 + 2x - 3 = 0."
                4. **Time Plugin**:
                - Purpose: Provide current date, time, or timezone-based calculations.
                - Examples: "What time is it in Tokyo?" or "What's the current date?"

                ### Classification Guidelines:
                - **`general_chat`**:
                - For casual conversation, greetings, or non-specific queries.
                - Examples:
                    - "Hi there!"
                - **`document_retrieval`**:
                - For requests involving finding, searching, or retrieving documents or information of hotels
                - **`plugin_usage`**:
                - For tasks requiring plugins such as WeatherAPI, Math, or Time.
                - When the plugin usage involves:
                    - **WeatherAPI Plugin**: Extract the city or location from the input. If no location is mentioned, return `"city": null`.
                    - **Math Plugin**: Identify mathematical operations or queries to solve.
                    - **Time Plugin**: Handle time or date-related questions.

                - **`unknown`**:
                - If you cannot confidently classify the input into one of the above categories.

                ### Response Requirements:
                - Always return a valid JSON object in this format:
                {
                    "intent": "string",  // One of: 'general_chat', 'document_retrieval', 'plugin_usage', or 'unknown'.
                    "confidence": float,  // A confidence score between 0.0 and 1.0.
                    "plugin": "string",  // The plugin to use if intent is 'plugin_usage'. One of: 'weather', 'math', 'time', or 'unknown'.
                    "city": "string"  // Extracted city/location if applicable; otherwise, return null.
                }

                ### Important Notes:
                - Adhere strictly to the JSON schema. Do not include explanations or additional text outside the JSON object.
                - Provide accurate and confident classifications. If uncertain, default to:

                Now, classify the following user input:
                """,
            ),
            ChatMessageContent(
                role="user",
                content=f"""
                "{input}"
                """,
            ),
        ]


        logger.debug("Step 3: Chat history prepared and sent to LLM: %s", [msg.content for msg in history])

        try:
            logger.info("Step 4: Invoking CustomChatCompletionAgent for intent detection.")
            async for response in self.agent.invoke_with_validation(history):
                logger.debug("Step 5: Response received from agent: %s", response.content)

                # Log raw LLM response
                raw_content = response.content.strip()
                logger.info("Raw LLM response: %s", raw_content)

                # Handle Markdown-style formatting (if applicable)
                if raw_content.startswith("```") and raw_content.endswith("```"):
                    logger.debug("Stripping Markdown-style backticks from response.")
                    raw_content = raw_content[3:-3].strip()

                # Handle LLM error responses
                if "Error:" in raw_content:
                    logger.error("LLM returned an error message: %s", raw_content)
                    continue  # Skip this iteration and retry if applicable

                # Parse the cleaned response
                try:
                    intent_data = json.loads(raw_content)
                    intent = intent_data.get("intent", "unknown")
                    confidence = intent_data.get("confidence", 0.0)
                    plugin = intent_data.get("plugin", "unknown")
                    city = intent_data.get("city", None)

                    # Validate response format
                    if intent not in {"general_chat", "document_retrieval", "plugin_usage", "unknown"} or not isinstance(confidence, (int, float)):
                        raise ValueError("Response does not conform to the expected schema.")

                    logger.info("Step 6: Parsed intent: %s with confidence: %s", intent, confidence)
                    logger.info("Plugin: %s, City: %s", plugin, city)

                    # Redirect unknown intents to general chat
                    if intent == "unknown":
                        logger.warning("Redirecting unknown intent to 'general_chat'.")
                        intent = "general_chat"
                        confidence = 0.5  # Assign a moderate confidence level

                    return json.dumps({"intent": intent, "confidence": confidence, "plugin": plugin, "city": city})

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error("Failed to parse or validate JSON response: %s", e)
                    logger.debug("Raw response content: %s", raw_content)
                    continue  # Retry if applicable

        except Exception as e:
            logger.error("Error during LLM invocation: %s", e, exc_info=True)
            return json.dumps({"intent": "general_chat", "confidence": 0.5, "plugin": "unknown", "city": None})
