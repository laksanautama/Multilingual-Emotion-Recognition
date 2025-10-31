import sys
from utils import load_environment_variables
from crosslingual_ER.scripts.run_model_comparion import run
from llm_evaluation.llm_code.few_shot_adaptation.run_few_shot import run_fs
import argparse
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
    print("1: Perform Few-Shot Adaptation")
    print("2: Perform LoRA Fine-Tuning and Evaluation")
    print("3: Perform RAG Pipeline Evaluation")
    print("4: Go back to Main Menu")

    
    choice = input("Select a method number (1, 2, or 3): ")
    return choice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_name', type=str, default='indolem/indobert-base-uncased')
    parser.add_argument('--llm_name', type=str, default='gemini-2.5-flash')
    # parser.add_argument('--adaptation_method', type=str, default='few-shot')
    args = parser.parse_args()


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
                    print("Running Few-Shot Adaptation...")
                    run_fs(keys, args.llm_name)
                elif task2_choice == '2':
                    print("Running LoRA Adaptation Method...")
                    # run_model_comparison.run_rag_evaluation(keys)
                elif task2_choice == '3':
                    print("Running RAG Adaptation Method ...")
                elif task2_choice == '4':
                    print("Returning to Main Menu...")
                    break
                else:
                    print("Invalid choice. Please select a valid method number.")

        elif choice == '3':
            print("Exiting the program.")
            sys.exit(0)
        else:
            print("Invalid choice. Please select a valid task number.")


if __name__ == "__main__":
    main()