import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

# Azure Cognitive Search Configuration
AZURE_COGNITIVE_SEARCH_ENDPOINT = os.getenv("AZURE_COGNITIVE_SEARCH_ENDPOINT", "")
AZURE_COGNITIVE_SEARCH_ADMIN_KEY = os.getenv("AZURE_COGNITIVE_SEARCH_ADMIN_KEY", "")
AZURE_COGNITIVE_SEARCH_INDEX_NAME = os.getenv("AZURE_COGNITIVE_SEARCH_INDEX_NAME", "")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # Logging level, e.g., DEBUG, INFO, ERROR

# Validation of Mandatory Configurations
required_configs = {
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_COGNITIVE_SEARCH_ENDPOINT": AZURE_COGNITIVE_SEARCH_ENDPOINT,
    "AZURE_COGNITIVE_SEARCH_ADMIN_KEY": AZURE_COGNITIVE_SEARCH_ADMIN_KEY,
}

missing_configs = [key for key, value in required_configs.items() if not value]
if missing_configs:
    raise EnvironmentError(
        f"Missing mandatory configuration(s): {', '.join(missing_configs)}. Please check your .env file."
    )
