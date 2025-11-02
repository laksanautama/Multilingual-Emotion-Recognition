import argparse
from utils import load_json_file
from crosslingual_ER.scripts.model_configs import DATA_CONFIG
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss, jaccard_score
)


def evaluate_fs():
    """Function to evaluate few-shot adaptation results."""
    print("Evaluating Few-Shot Adaptation Results...")

    emotion_results = load_json_file("few_shot_results", 'few_shot')
    sumscore = 0
    for emotion, results in emotion_results.items():
        f1 = f1_score(results["ground_truths"], results["predictions"], average='macro')
        print(f"F1 Score for {emotion}: {f1}")
        sumscore += f1
    print(f"Average F1 Score across all emotions: {sumscore / DATA_CONFIG['NUM_LABELS']}")


def evaluate_rag():
    """Function to evaluate RAG adaptation results."""
    print("Evaluating RAG Adaptation Results...")

    column_names = DATA_CONFIG["LABELS"].append("no emotion")
    mlb = MultiLabelBinarizer(classes=column_names)
    mlb.fit([[]])
    rag_results = load_json_file("rag_evaluation_results", 'rag')
    y_true = mlb.fit_transform(rag_results["ground_truths"])
    y_pred = mlb.fit_transform(rag_results["predictions"])    
    print("Exact Match Ratio:", accuracy_score(y_true, y_pred))
    print("Hamming Loss:", hamming_loss(y_true, y_pred))
    print("Micro-F1:", f1_score(y_true, y_pred, average='micro'))
    print("Macro-F1:", f1_score(y_true, y_pred, average='macro'))
    print("Jaccard (samples):", jaccard_score(y_true, y_pred, average='samples'))


        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['evaluate_fs', 'evaluate_lora', 'evaluate_rag'], help='Task to perform')
    args = parser.parse_args()

    if args.task == 'evaluate_fs':
        evaluate_fs()
    elif args.task == 'evaluate_rag':
        evaluate_rag()

if __name__ == "__main__":
    main()
