import os

def load_prompt(filename: str) -> str:
    """
    Loads a prompt template from the 'prompts' folder.

    Args:
        filename (str): Name of the file containing the prompt.

    Returns:
        str: The content of the prompt file as a string.
    """
    # Get the absolute path to the project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    prompts_dir = os.path.join(base_dir, "prompts")
    filepath = os.path.join(prompts_dir, filename)

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{filename}' not found in {prompts_dir}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the prompt file: {str(e)}")
