import os
import sys
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

def task_selection():
    """Displaying the task menu."""
    print("\n--- Balinesse Language Emotion Recognition (BaLER)---")
    print("1: Crosslingual Emotion Recognition: Using Pre-trained Language Model trained on familiy languages")
    print("2: LLM Emotion Recognition: Using various LLMs with different adaptation methods (LoRA & RAG)")
    print("3: Exit")
    
    choice = input("Select a task number (1, 2, or 3): ")
    return choice

if __name__ == "__main__":
    keys = load_environment_variables()
    
    while True:
        choice = task_selection()
        
        if choice == '1':
            print("Running Crosslingual Emotion Recognition...")
            """todo"""
        elif choice == '2':
            print("Running LLM Emotion Recognition...")
            """todo"""
        elif choice == '3':
            print("Exiting the program.")
            sys.exit(0)
        else:
            print("Invalid choice. Please select a valid task number.")