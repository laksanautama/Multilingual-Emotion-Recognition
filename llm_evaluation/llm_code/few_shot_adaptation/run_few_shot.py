# from llm_model_selection import llm_selection
from llm_evaluation.llm_code.llm_utility.llm_utility import llm_selection, classify_text, select_language_config
from utils import llm_dataset_preparation, create_train_examples, create_val_examples, save_results_to_file, save_analysis_results
from utils import DATA_CONFIG
from ..llm_utility.llm_config import PROMPT_CONFIG
import logging
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate




def run_fs(keys: dict, llm_model_name: str, prompt_language: str):
    
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    """Placeholder function for running few-shot adaptation tasks."""
    print("-" * 20)
    print("Balinese Text Emotion Recognition using Few-Shot Adaptation")
    print("-" * 20)

    llm_model = llm_selection(llm_model_name, keys)
    train_data, val_data = llm_dataset_preparation(DATA_CONFIG["TARGET_TEST_FILENAME"], DATA_CONFIG["TRAIN_SPLIT_SIZE"])
    val_data_samples = create_val_examples(val_data, DATA_CONFIG["NUM_VAL_SAMPLES"])

    emotion_results = {}
    emotion_analysis = {}
    
    config = select_language_config(prompt_language)
    
    for emotion in DATA_CONFIG["LABELS"]:
        print(f"\nAdapting and Evaluating for Emotion: {emotion}")
        train_examples = create_train_examples(train_data, emotion, DATA_CONFIG["NUM_TRAIN_SAMPLES_PER_CLASS"], prompt_language)
        fs_prompts = FewShotPromptTemplate(
                    examples=train_examples,
                    example_prompt = PromptTemplate(
                                    input_variables=[config['input'], config['answer']],
                                    template=config['example_template']
                    ),
                    prefix=config['prefix'].format(emotion=emotion), 
                    suffix=config['suffix'],
                    input_variables=["input"],
                    example_separator="\n"
            )
        chain = fs_prompts | llm_model
        predictions = []
        ground_truths = []
        text = []
        justification = []

        for i in range(len(val_data_samples)):
            result, output_text = classify_text(val_data_samples['sentence'].iloc[i], chain, prompt_language)
            ground_truth = 'yes' if val_data_samples[emotion].iloc[i] == 1 else 'no'
            print(f"Input: {val_data_samples['sentence'].iloc[i]}")
            print(f"Prediction: {result}")
            print(f"Ground Truth: {ground_truth}")
            print("+"*20)
            predictions.append(result)
            ground_truths.append(ground_truth)
            text.append(val_data_samples[emotion].iloc[i])
            justification.append(output_text)
        

        
        emotion_results[emotion] = {"predictions": predictions, 
                                    "ground_truths": ground_truths
                                    }
        
    emotion_analysis[prompt_language] = {'text': text,
                                         'justification': justification}

    results_filename = "few_shot_results"
    mer_analysis_filename = "few_shot_analysis"
    save_results_to_file(emotion_results, results_filename, 'few_shot', prompt_language)
    save_analysis_results(emotion_analysis, mer_analysis_filename, 'few_shot', prompt_language)
    