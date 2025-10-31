import argparse
from utils import load_json_file
from crosslingual_ER.scripts.model_configs import DATA_CONFIG
from sklearn.metrics import f1_score

def evaluate_fs():
    """Function to evaluate few-shot adaptation results."""
    print("Evaluating Few-Shot Adaptation Results...")

    emotion_results = load_json_file("few_shot_results")
    sumscore = 0
    for emotion, results in emotion_results.items():
        f1 = f1_score(results["ground_truths"], results["predictions"], average='macro')
        print(f"F1 Score for {emotion}: {f1}")
        sumscore += f1
    print(f"Average F1 Score across all emotions: {sumscore / DATA_CONFIG['NUM_LABELS']}")


    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['evaluate_fs', 'evaluate_lora', 'evaluate_rag'], help='Task to perform')
    args = parser.parse_args()

    if args.task == 'evaluate_fs':
        evaluate_fs()

if __name__ == "__main__":
    main()
