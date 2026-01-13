from utils import check_faiss_exists, llm_dataset_preparation, create_val_examples, save_results_to_file, save_analysis_results, load_target_test_data, load_huggingface_dataset
from utils import llm_selection, select_language_config, strip_result_content
from utils import DATA_CONFIG
from utils import DIRECTORY_PATH, PROMPT_CONFIG
from llm_evaluation import store_vectors_database, rag_retriever, rag_classifier, bin_rag_classifier
from langchain_core.prompts import PromptTemplate
import json
import os
import argparse
import pandas as pd
from utils import load_environment_variables

def run_rag(keys: dict, llm_model_name: str, prompt_language: str):
    print("-" * 20)
    print("Balinese Text Emotion Recognition using RAG Adaptation Method")
    print("-" * 20)

    MAIN_DIR = os.path.dirname(__file__)
    results_dir = os.path.join(MAIN_DIR, DIRECTORY_PATH["RAG_RESULTS_DIR"])
    filepath = os.path.join(MAIN_DIR,
                            'crosslingual_ER',
                            DATA_CONFIG["TEST_DATA_DIR"],
                            DATA_CONFIG["TARGET_TEST_FILENAME"]
                            )
    val_data = pd.read_csv(filepath)
    train_data, __ = load_huggingface_dataset(DATA_CONFIG["SOURCE_HF_DATASET"], DATA_CONFIG["DATASET_LANGUAGES"], 'train', keys)
    val_samples = create_val_examples(val_data, DATA_CONFIG["NUM_VAL_SAMPLES"])

    if check_faiss_exists('index') is False:
        store_vectors_database(train_data, keys)
    
    config = select_language_config(prompt_language)
    llm = llm_selection(llm_model_name, keys)
    retriever = rag_retriever(llm_model_name, keys, k = 50)
  
    
    classification_prompt = PromptTemplate(
        template=config['clf_template'],
        input_variables=["context", "query", "target_labels"]
    )
    chain = classification_prompt | llm

    emotions = DATA_CONFIG["LABELS"]
    lang = prompt_language

    emotion_results = {}
    emotion_analysis = {}
    emotion_justification = {}


    for emo in emotions:
        print(f"Target Emotion for Retrieval: {emo}")
   
        predictions = []
        ground_truths = []
        justification = []

        for _, row in val_samples.iterrows():
            sentence = row['sentence']
            val_true = 'yes' if row[emo] == 1 else 'no'

            

            out = bin_rag_classifier(
            query=sentence,
            retriever=retriever,
            chain=chain,
            target_emotion=emo,
            lang=lang
            )

            content = out.content.lower()
            answer, reason = strip_result_content(
            content, 
            config['answer'], 
            config['reason']
            )
            print(f"Input: {sentence}")
            print(f"Prediction for {emo}: {answer}")
            print(f"Ground Truth for {emo}: {val_true}")
            print(f"Reason: {reason}")
            print("-" * 15)

            predictions.append(answer)
            ground_truths.append(val_true)
            justification.append(reason)
        
        emotion_justification[emo] = justification
        
        emotion_results[emo] = {"predictions": predictions, 
                                    "ground_truths": ground_truths
                                    }
    
    emotion_analysis[llm_model_name] = {'text': list(val_samples['sentence']),
                                         'justification': justification}


    print("Evaluation completed.")
    print("Saving results...")

    results_filename = "rag_evaluation_results"
    mer_analysis_filename = "rag_analysis"
    save_results_to_file(emotion_results, results_filename, 'rag', llm_model_name)
    save_analysis_results(emotion_analysis, mer_analysis_filename, 'rag', llm_model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, default='gemini-2.5-flash')
    parser.add_argument('--prompt_language', type=str, default='english', help='Language for the prompt (english/indonesian/balinese)')
    args = parser.parse_args()

    keys = load_environment_variables()
    run_rag(keys, args.llm_name, args.prompt_language)
