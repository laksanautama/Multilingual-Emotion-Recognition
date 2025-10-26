import os
from dotenv import load_dotenv

def load_environment_variables():
    """Load environment variables from a .env file."""
    load_dotenv()
    keys = {
        "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
        "GEMINI_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY")
    }
    missing_keys = [key for key, value in keys.items() if value is None]
    if missing_keys:
        print(f"Warning: Missing API/TOKEN keys for: {', '.join(missing_keys)}")
    return keys