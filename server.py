import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.connectors.memory.azure_cognitive_search import (
    AzureCognitiveSearchMemoryStore,
)
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.core_plugins.http_plugin import HttpPlugin
from semantic_kernel.functions.kernel_arguments import KernelArguments

from agents.custom_chat_completion_agent import CustomChatCompletionAgent
from agents.custom_retrieval_agent import CustomRetrievalAgent
from plugins.intent_detection import IntentDetectionPlugin
from utils.config import (
    AZURE_COGNITIVE_SEARCH_ADMIN_KEY,
    AZURE_COGNITIVE_SEARCH_ENDPOINT,
    AZURE_COGNITIVE_SEARCH_INDEX_NAME,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_ENDPOINT,
)
from utils.logger import setup_logging

logger = setup_logging(log_level="DEBUG", log_to_file=True)

# Initialize global variables
SK_KERNEL = None
CHAT_AGENT = None
RETRIEVAL_AGENT = None


class ChatRequest(BaseModel):
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SK_KERNEL, CHAT_AGENT, RETRIEVAL_AGENT
    try:
        logger.info("Initializing Semantic Kernel and agents...")
        SK_KERNEL, CHAT_AGENT, RETRIEVAL_AGENT = await initialize_kernel_and_agents()
        logger.info("Agents initialized successfully.")
        yield
    except Exception as e:
        logger.critical("Error during lifespan initialization: %s", e, exc_info=True)
        raise
    finally:
        logger.info("Cleaning up resources...")


app = FastAPI(lifespan=lifespan)


async def initialize_kernel_and_agents():
    logger.info("Initializing Semantic Kernel...")
    kernel = Kernel()

    try:
        # Configure Azure OpenAI Services
        logger.debug("Setting up Azure OpenAI chat and embedding services...")
        chat_service = AzureChatCompletion(
            service_id="chat_service",
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        )
        kernel.add_service(chat_service)
        logger.info("AzureChatCompletion registered successfully as 'chat_service'.")

        embedding_service = AzureTextEmbedding(
            service_id="embedding_service",
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        )
        kernel.add_service(embedding_service)
        logger.info(
            "AzureTextEmbedding registered successfully as 'embedding_service'."
        )

        logger.info("Azure OpenAI services configured successfully.")

        # Configure Azure Cognitive Search
        logger.debug("Setting up Azure Cognitive Search...")
        acs_connector = AzureCognitiveSearchMemoryStore(
            vector_size=1536,
            search_endpoint=AZURE_COGNITIVE_SEARCH_ENDPOINT,
            admin_key=AZURE_COGNITIVE_SEARCH_ADMIN_KEY,
        )
        await acs_connector.create_collection(
            collection_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME
        )
        acs_connector.service_id = "memory_service"
        kernel.add_service(acs_connector)
        logger.info(
            "Azure Cognitive Search registered successfully as 'memory_service'."
        )

        # Debug Kernel services after registration
        logger.debug("Current Kernel services after registration: %s", kernel.services)

        # Initialize Custom ChatCompletionAgent
        logger.debug("Initializing CustomChatCompletionAgent...")
        chat_agent = CustomChatCompletionAgent(
            kernel=kernel, instructions="Be a helpful assistant."
        )
        logger.info("CustomChatCompletionAgent initialized successfully.")

        # Register the IntentDetection plugin
        logger.debug("Registering IntentDetection plugin...")
        intent_detection_plugin = IntentDetectionPlugin(agent=chat_agent)
        kernel.add_plugin(intent_detection_plugin, plugin_name="intent_detection")
        logger.info("IntentDetectionPlugin registered successfully.")

        # Initialize Custom Retrieval Agent
        logger.debug("Initializing CustomRetrievalAgent...")
        retrieval_agent = CustomRetrievalAgent(
            search_endpoint=AZURE_COGNITIVE_SEARCH_ENDPOINT,
            api_key=AZURE_COGNITIVE_SEARCH_ADMIN_KEY,
            index_name=AZURE_COGNITIVE_SEARCH_INDEX_NAME,
            vector_field_name="embedding",  # Ensure vector field alignment
        )
        logger.info(
            "CustomRetrievalAgent initialized successfully with extended functionality."
        )

        # Add HttpPlugin for weather
        logger.debug("Registering HttpPlugin for weather...")
        http_plugin = HttpPlugin()
        kernel.add_plugin(http_plugin, "http")
        logger.info("HttpPlugin registered successfully.")

        logger.info("Semantic Kernel initialization completed successfully.")
        return kernel, chat_agent, retrieval_agent

    except Exception as e:
        logger.critical(
            "Error during Semantic Kernel initialization: %s", e, exc_info=True
        )
        raise RuntimeError("Failed to initialize Semantic Kernel and agents.") from e


@app.post("/chat", response_model=dict)
async def unified_chat_endpoint(request: ChatRequest):
    global CHAT_AGENT, RETRIEVAL_AGENT, SK_KERNEL
    user_input = request.message.strip()

    if not user_input:
        logger.warning("Received an empty user input.")
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        logger.info("Starting unified chat processing for input: %s", user_input)

        # Step 1: Detect intent
        logger.info("Detecting intent for user input...")
        intent_arguments = KernelArguments()
        intent_arguments["input"] = user_input

        logger.debug("Prepared KernelArguments for intent detection: %s", intent_arguments)
        intent_detection_response = await SK_KERNEL.invoke(
            function_name="DetectIntent",
            plugin_name="intent_detection",
            arguments=intent_arguments,
        )

        logger.debug(f"Raw intent detection response object: {intent_detection_response}")
        if not hasattr(intent_detection_response, "value") or intent_detection_response.value is None:
            logger.error("intent_detection_response.value is None. Returning fallback response.")
            return {"intent": "unknown", "confidence": 0.0, "plugin": "unknown", "city": None}

        try:
            intent_data = json.loads(intent_detection_response.value)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            return {"intent": "unknown", "confidence": 0.0, "plugin": "unknown", "city": None}

        logger.debug(f"Intent detection parsed response: {intent_data}")
        intent = intent_data.get("intent", "unknown")
        confidence = intent_data.get("confidence", 0.0)
        plugin = intent_data.get("plugin", "unknown")
        city = intent_data.get("city", None)
        logger.info(f"Detected intent: {intent} with confidence {confidence}, plugin: {plugin}, city: {city}")

        # Step 2: Route the request based on the intent
        if intent == "general_chat":
            logger.info("Routing to chat agent...")
            history = [ChatMessageContent(role="user", content=user_input)]
            logger.debug("Chat history for chat agent: %s", history)
            async for response in CHAT_AGENT.invoke_with_validation(history):
                logger.debug("Chat agent response: %s", response.content)
                return {"response": response.content}

        elif intent == "document_retrieval":
            logger.info("Routing to retrieval agent...")
            results = await RETRIEVAL_AGENT.retrieve_by_text(
                query_text=user_input, top_k=5
            )
            logger.debug("Document retrieval results: %s", results)
            return {"response": results}

        elif intent == "plugin_usage":
            logger.info("Handling plugin usage...")
            if plugin == "weather" and city:
                logger.debug("Detected weather-related query.")
                url = f"http://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q={city}"
                logger.debug("Weather API URL: %s", url)
                weather_data = await SK_KERNEL.plugins["http"].get(url)
                logger.debug("Weather data response: %s", weather_data)
                return {"response": weather_data}

            elif plugin == "math":
                logger.info("Routing to math plugin...")
                # Example: Perform math operations based on user_input
                math_response = await SK_KERNEL.plugins["math"].calculate(user_input)
                logger.debug("Math plugin response: %s", math_response)
                return {"response": math_response}

            elif plugin == "time":
                logger.info("Routing to time plugin...")
                time_response = await SK_KERNEL.plugins["time"].get_current_time()
                logger.debug("Time plugin response: %s", time_response)
                return {"response": time_response}

            else:
                logger.warning("Unrecognized plugin usage or insufficient data: %s", intent_data)
                return {"response": "Plugin functionality not recognized or insufficient data provided."}

        else:
            logger.warning(f"Unrecognized intent: {intent}")
            return {"response": "I’m not sure how to help with that."}

    except Exception as e:
        logger.error("Error during unified chat processing: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.post("/detect_intent", response_model=dict)
async def detect_intent_endpoint(request: ChatRequest):
    global SK_KERNEL
    user_input = request.message.strip()
    if not user_input:
        logger.warning("Received an empty user input for intent detection.")
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        logger.info("Processing intent detection for input: '%s'", user_input)

        # Prepare KernelArguments
        arguments = KernelArguments()
        arguments["input"] = user_input

        logger.debug("Prepared KernelArguments for intent detection: %s", arguments)
        result = await SK_KERNEL.invoke(
            function_name="DetectIntent",
            plugin_name="intent_detection",
            arguments=arguments,
        )

        intent_data = json.loads(result.value) if hasattr(result, "value") else {}
        logger.debug("Intent detection result: %s", intent_data)

        intent = intent_data.get("intent", "unknown")
        confidence = intent_data.get("confidence", 0.0)
        logger.info("Intent detected: %s with confidence: %s", intent, confidence)
        return {"intent": intent}

    except Exception as e:
        logger.error("Error during intent detection: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/retrieval", response_model=dict)
async def retrieval_endpoint(request: ChatRequest):
    global RETRIEVAL_AGENT
    user_input = request.message.strip()
    if not user_input:
        logger.warning("Received an empty user input for document retrieval.")
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        logger.info("Processing document retrieval for input: '%s'", user_input)

        results = await RETRIEVAL_AGENT.retrieve_by_text(query_text=user_input, top_k=5)
        logger.debug("Document retrieval results: %s", results)
        return {"results": results}

    except Exception as e:
        logger.error("Error during document retrieval: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post("/test_llm", response_model=dict)
async def test_llm_endpoint(request: ChatRequest):
    global CHAT_AGENT
    user_input = request.message.strip()

    history = [
        ChatMessageContent(
            role="system",
            content="You are an assistant. Analyze the input and classify it."
        ),
        ChatMessageContent(
            role="user",
            content=f"Classify this input: \"{user_input}\""
        ),
    ]
    logger.debug("History sent to LLM: %s", history)

    try:
        async for response in CHAT_AGENT.invoke_with_validation(history):
            logger.debug("LLM response: %s", response.content)
            return {"response": response.content}
    except Exception as e:
        logger.error("Error during LLM invocation: %s", e, exc_info=True)
        return {"error": str(e)}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
