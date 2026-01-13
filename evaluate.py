import argparse
from utils import load_json_file, get_folder_name, translate_label
from utils import DATA_CONFIG
from utils import DIRECTORY_PATH
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss, jaccard_score
)


def evaluate_score(task: str):
    if task == 'evaluate_fs':
        dir_path = DIRECTORY_PATH["FEW_SHOT_RESULTS_DIR"]
        result_type = 'few_shot'
        filename = "few_shot_results"
    elif task == 'evaluate_rag':
        dir_path = DIRECTORY_PATH["RAG_RESULTS_DIR"]
        result_type = 'rag'
        filename = "rag_results"
    elif task == 'evaluate_lora':
        dir_path = DIRECTORY_PATH["LORA_RESULTS_DIR"]
        result_type = 'lora'
        filename = "lora_results"
    elif task == 'evaluate_zero_shot':
        dir_path = DIRECTORY_PATH["ZERO_SHOT_RESULTS_DIR"]
        result_type = 'zero_shot'
        filename = "zero_shot_results"
    elif task == 'evaluate_multiagents':
        dir_path = DIRECTORY_PATH["MULTIAGENTS_RESULT_DIR"]
        result_type = 'multiagents'
        filename = 'multiagents_results-50'

    llm_model = get_folder_name(dir_path)
    for model in llm_model:
        print(f"\n--- Evaluating for LLM Model: {model} ---")
        emotion_results = load_json_file(filename, result_type, model)
        sumscore = 0
        for emotion, results in emotion_results.items():
            # f1 = f1_score(results["ground_truths"], results["predictions"], average='macro')
            f1 = f1_score(results["ground_truths"], results["predictions"], average='macro')
            print(f"F1 Score for {emotion}: {f1}")
            sumscore += f1
        print(f"Average F1 Score across all emotions: {sumscore / DATA_CONFIG['NUM_LABELS']} in LLM model: {model}")


    
def evaluate_fs():
    """Function to evaluate few-shot adaptation results."""
    print("Evaluating Few-Shot Adaptation Results...")
    dir_path = DIRECTORY_PATH["FEW_SHOT_RESULTS_DIR"]
    prompt_lang = get_folder_name(dir_path)
    
    for lang in prompt_lang:
        print(f"\n--- Evaluating for Prompt Language: {lang} ---")

        emotion_results = load_json_file("few_shot_results", 'few_shot', lang)
        sumscore = 0
        for emotion, results in emotion_results.items():
            f1 = f1_score(results["ground_truths"], results["predictions"], average='macro')
            print(f"F1 Score for {emotion}: {f1}")
            sumscore += f1
        print(f"Average F1 Score across all emotions: {sumscore / DATA_CONFIG['NUM_LABELS']} in language: {lang}")


def evaluate_rag():
    """Function to evaluate RAG adaptation results."""
    print("Evaluating RAG Adaptation Results...")
    """
    column_names = DATA_CONFIG["LABELS"]
    column_names.append('no emotion')
    print(f"Column Names for Binarizer: {column_names}")

    dir_path = DIRECTORY_PATH["RAG_RESULTS_DIR"]
    prompt_lang = get_folder_name(dir_path)
    mlb_en = MultiLabelBinarizer(classes=column_names)
    mlb_en.fit([column_names])

    for lang in prompt_lang:
        print(f"\n--- Evaluating for Prompt Language: {lang} ---")
        lang_col_names = translate_label(column_names, lang)
        mlb = MultiLabelBinarizer(classes=lang_col_names)
        mlb.fit([lang_col_names])
        rag_results = load_json_file("rag_evaluation_results", 'rag', lang)
        truths = rag_results['ground_truths']
        preds = rag_results['predictions']
        
        y_true = mlb_en.transform(truths)
        y_pred = mlb.transform(preds)
        # print(f"Ground truth: {y_true}")
        # print(f"Predictions: {y_pred}")   
        print("Exact Match Ratio:", accuracy_score(y_true, y_pred))
        print("Hamming Loss:", hamming_loss(y_true, y_pred))
        print("Micro-F1:", f1_score(y_true, y_pred, average='micro'))
        print("Macro-F1:", f1_score(y_true, y_pred, average='macro'))
        print("Jaccard (samples):", jaccard_score(y_true, y_pred, average='samples'))
        """
    dir_path = DIRECTORY_PATH["RAG_RESULTS_DIR"]
    prompt_lang = get_folder_name(dir_path)
    for lang in prompt_lang:
        print(f"\n--- Evaluating for Prompt Language: {lang} ---")

        emotion_results = load_json_file("rag_evaluation_results", 'rag', lang)
        sumscore = 0
        for emotion, results in emotion_results.items():
            f1 = f1_score(results["ground_truths"], results["predictions"], average='macro')
            print(f"F1 Score for {emotion}: {f1}")
            sumscore += f1
        print(f"Average F1 Score across all emotions: {sumscore / DATA_CONFIG['NUM_LABELS']} in language: {lang}")



        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['evaluate_fs', 'evaluate_lora', 'evaluate_rag', 'evaluate_zero_shot', 'evaluate_multiagents'], help='Task to perform')
    # parser.add_argument('--prompt_language', type=str, default='english', help='Language for the prompt (english/indonesian/balinese)')
    args = parser.parse_args()
    evaluate_score(args.task)
    # if args.task == 'evaluate_fs':
    #     evaluate_fs()
    # elif args.task == 'evaluate_rag':
    #     evaluate_rag()

if __name__ == "__main__":
    main()
