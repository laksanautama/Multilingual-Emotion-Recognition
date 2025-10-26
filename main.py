import sys
from utils.api_key_handler import load_environment_variables
from crosslingual_ER.scripts.run_model_comparion import run
# from crosslingual_ER.scripts.run_model_comparion import run_model_comparison 


def task_selection():
    """Displaying the task menu."""
    print("\n--- Balinesse Language Emotion Recognition (BaLER)---")
    print("1: Crosslingual Emotion Recognition: Using Pre-trained Language Model trained on familiy languages")
    print("2: LLM Emotion Recognition: Using various LLMs with different adaptation methods (LoRA & RAG)")
    print("3: Exit")
    
    choice = input("Select a task number (1, 2, or 3): ")
    return choice

def task_2_selection():
    """Presents the sub-menu for LLM Evaluation methods."""
    print("\n--- Task 2: LLM Emotion Recognition Sub-Menu ---")
    print("1: Perform LoRA Fine-Tuning and Evaluation")
    print("2: Perform RAG Pipeline Evaluation")
    print("3: Go back to Main Menu")
    
    choice = input("Select a method number (1, 2, or 3): ")
    return choice

if __name__ == "__main__":
    keys = load_environment_variables()
    
    while True:
        choice = task_selection()
        
        if choice == '1':
            print("Running Crosslingual Emotion Recognition...")
            run(keys)
        elif choice == '2':
            while True:
                task2_choice = task_2_selection()
                if task2_choice == '1':
                    print("Running LoRA Fine-Tuning and Evaluation...")
                    # run_model_comparison.run_lora_finetuning_evaluation(keys)
                elif task2_choice == '2':
                    print("Running RAG Pipeline Evaluation...")
                    # run_model_comparison.run_rag_evaluation(keys)
                elif task2_choice == '3':
                    break
                else:
                    print("Invalid choice. Please select a valid method number.")

        elif choice == '3':
            print("Exiting the program.")
            sys.exit(0)
        else:
            print("Invalid choice. Please select a valid task number.")