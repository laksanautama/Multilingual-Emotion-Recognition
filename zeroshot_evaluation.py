from utils import llm_selection, classify_text, select_language_config
from utils import llm_dataset_preparation, create_train_examples, create_val_examples, save_results_to_file, save_analysis_results, load_target_test_data, load_huggingface_dataset
from utils import DATA_CONFIG, PROMPT_CONFIG
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import argparse
from utils import load_environment_variables

def run_zs(keys: dict, llm_model_name: str, prompt_language: str, evaluation: bool = True):
    
    """Placeholder function for running few-shot adaptation tasks."""
    print("-" * 20)
    print("Balinese Text Emotion Recognition using Zero-Shot Adaptation")
    print("-" * 20)

    mdl = llm_selection(keys, llm_model_name)
    llm_model = mdl[llm_model_name]
    val_data = load_target_test_data(DATA_CONFIG["TARGET_TEST_FILENAME"], cross_lingual=False)
    # train_data, __ = load_huggingface_dataset(DATA_CONFIG["SOURCE_HF_DATASET"], DATA_CONFIG["DATASET_LANGUAGES"], 'train', keys)
    val_data_samples = create_val_examples(val_data, DATA_CONFIG["NUM_VAL_SAMPLES"])

    emotion_results = {}
    emotion_analysis = {}
    emotion_justification = {}

    
    config = select_language_config(prompt_language)
    
    for emotion in DATA_CONFIG["LABELS"]:
        print(f"\nEvaluating for Emotion: {emotion}")
        # train_examples = create_train_examples(train_data, emotion, DATA_CONFIG["NUM_TRAIN_SAMPLES_PER_CLASS"], prompt_language)
        """
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
        """
        prompt = ChatPromptTemplate.from_template(
            config['zero_shot_prefix'] + config['suffix']
        )
        chain = prompt | llm_model
        
        predictions = []
        ground_truths = []
        justification = []

        for i in range(len(val_data_samples)):
            result, output_text = classify_text(val_data_samples['sentence'].iloc[i], chain, prompt_language, emotion)
            ground_truth = 'yes' if val_data_samples[emotion].iloc[i] == 1 else 'no'
            print(f"Input: {val_data_samples['sentence'].iloc[i]}")
            print(f"Prediction: {result}")
            print(f"Ground Truth: {ground_truth}")
            print("+"*20)
            predictions.append(result)
            ground_truths.append(ground_truth)
            justification.append(output_text)
        
        emotion_justification[emotion] = justification
        

        
        emotion_results[emotion] = {"predictions": predictions, 
                                    "ground_truths": ground_truths
                                    }
        
    emotion_analysis[llm_model_name] = {'text': list(val_data_samples['sentence']),
                                         'justification': emotion_justification}

    results_filename = "zero_shot_results"
    mer_analysis_filename = "zero_shot_analysis"
    save_results_to_file(emotion_results, results_filename, 'zero_shot', llm_model_name)
    save_analysis_results(emotion_analysis, mer_analysis_filename, 'zero_shot', llm_model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_model_name', type=str, default='gemini-2.5-flash')
    parser.add_argument('--prompt_language', type=str, default='english', help='Language for the prompt (english/indonesian/balinese)')
    args = parser.parse_args()

    keys = load_environment_variables()
    run_zs(keys, args.llm_model_name, args.prompt_language, evaluation = True)
    