# from llm_model_selection import llm_selection
from utils import llm_selection, classify_text, select_language_config, multiagents_classify_text
from utils import llm_dataset_preparation, create_train_examples, create_val_examples, save_results_to_file, save_analysis_results, load_target_test_data, load_huggingface_dataset
from utils import DATA_CONFIG, PROMPT_CONFIG
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
import argparse
from collections import Counter
from utils import load_environment_variables


def run_fs(keys: dict, prompt_language: str, evaluation: bool = True):
    
    """Placeholder function for running few-shot adaptation tasks."""
    print("-" * 20)
    print("Balinese Text Emotion Recognition using Multi-Agents-Few-Shot Adaptation")
    print("-" * 20)

    llm_model_list = llm_selection(keys)
    val_data = load_target_test_data(DATA_CONFIG["TARGET_TEST_FILENAME"], cross_lingual=False)
    train_data, __ = load_huggingface_dataset(DATA_CONFIG["SOURCE_HF_DATASET"], DATA_CONFIG["DATASET_LANGUAGES"], 'train', keys)
    val_data_samples = create_val_examples(val_data, DATA_CONFIG["NUM_VAL_SAMPLES"])

    emotion_results = {}
    emotion_analysis = {}
    emotion_justification = {}

    
    config = select_language_config(prompt_language)
    
    for emotion in DATA_CONFIG["LABELS"]:
        print(f"\nAdapting and Evaluating for Emotion: {emotion}")
        train_examples = create_train_examples(train_data, emotion, DATA_CONFIG["NUM_TRAIN_SAMPLES_PER_CLASS"], prompt_language)
        fs_prompts = FewShotPromptTemplate(
                    examples=train_examples,
                    example_prompt = PromptTemplate(
                                    input_variables=[config['input'], config['answer'], config['conclusion']],
                                    template=config['example_template']
                    ),
                    prefix=config['prefix'].format(emotion=emotion), 
                    suffix=config['suffix'],
                    input_variables=["input"],
                    example_separator="\n"
            )
        # chain = fs_prompts | llm_model
        
        predictions = []
        ground_truths = []
        justification = []
        winner = []
        initial_results = []
        mcp = []

        for i in range(len(val_data_samples)):
            result = multiagents_classify_text(val_data_samples['sentence'].iloc[i], llm_model_list, fs_prompts, emotion)
            ground_truth = 'yes' if val_data_samples[emotion].iloc[i] == 1 else 'no'
            print(f"Input: {val_data_samples['sentence'].iloc[i]}")
            print(f"Prediction: {result["final_answer"]}")
            print(f"Ground Truth: {ground_truth}")
            print("+"*20)
            predictions.append(result["final_answer"])
            ground_truths.append(ground_truth)
            justification.append(result["final_reason"])
            winner.append(result["winner_model"])
            initial_results.append(result["model_initial_answer"])  
            mcp.extend(result["models_change_perspective"])

        emotion_justification[emotion] = justification
        mcp_occurence = Counter(mcp)
        emotion_results[emotion] = {"predictions": predictions, 
                                    "ground_truths": ground_truths,
                                    "models_change_itsperspective": mcp_occurence,
                                    "winner_model": winner,
                                    "models_initial_answer": initial_results,
                                    "models_final_answer": result["model_revision"]
                                    }
        
    emotion_analysis["multiagents"] = {'text': list(val_data_samples['sentence']),
                                         'justification': emotion_justification}

    results_filename = "multiagents_results"
    mer_analysis_filename = "multiagents_analysis"
    save_results_to_file(emotion_results, results_filename, 'multiagents', 'multiagents')
    save_analysis_results(emotion_analysis, mer_analysis_filename, 'multiagents', 'multiagents')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--llm_model_name', type=str, default='gemini-2.5-flash')
    parser.add_argument('--prompt_language', type=str, default='english', help='Language for the prompt (english/indonesian/balinese)')
    args = parser.parse_args()

    keys = load_environment_variables()
    run_fs(keys, args.prompt_language, evaluation = True)
    