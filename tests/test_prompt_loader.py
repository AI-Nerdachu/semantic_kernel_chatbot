import os
import sys

# Add the src/ directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from utils.prompt_loader import load_prompt

def test_load_prompt():
    try:
        rag_prompt = load_prompt("rag_prompt.txt")
        print("RAG Prompt Loaded Successfully:")
        print(rag_prompt)

        json_schema_prompt = load_prompt("json_schema_prompt.txt")
        print("JSON Schema Prompt Loaded Successfully:")
        print(json_schema_prompt)

    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_load_prompt()
