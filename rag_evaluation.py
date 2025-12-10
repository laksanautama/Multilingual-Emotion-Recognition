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
    #train_data, val_data = llm_dataset_preparation(DATA_CONFIG["TARGET_TEST_FILENAME"], DATA_CONFIG["TRAIN_SPLIT_SIZE"])
    val_data = pd.read_csv(filepath)
    #val_data = load_target_test_data(DATA_CONFIG["TARGET_TEST_FILENAME"], cross_lingual=False)
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

    # results = []
    # ground_truths = []
    # val_text = []
    # label_result = []

    emotions = DATA_CONFIG["LABELS"]
    lang = prompt_language

    emotion_results = {}
    emotion_analysis = {}
    emotion_justification = {}


    for emo in emotions:
        print(f"Target Emotion for Retrieval: {emo}")
        # pos_retriever = rag_retriever(llm_model_name, keys, DATA_CONFIG["RAG_TOP_K"], emotion = emo, positive=True)
        # neg_retriever = rag_retriever(llm_model_name, keys, DATA_CONFIG["RAG_TOP_K"], emotion = emo, positive=False)

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


    """
    for _, row in val_samples.iterrows():
        print(row['sentence'])
        
        label = rag_classifier(
            query = row['sentence'],
            retriever = retriever,
            chain = chain,
            labels = DATA_CONFIG["LABELS"],
            lang = prompt_language
          )
        label_content = label.content.lower()
        print(f"Reason: {label_content}")
        print("-" * 15)

        answer, reason = strip_result_content(label_content, config['answer'], config['reason'])       
        val_data_emotions = row['emotions'].strip("[]").replace("'", "")
        emotion = 'no emotion' if val_data_emotions == '' else val_data_emotions

        clean = answer.split(", ")
        val_data_clean = emotion.split(", ")

        ground_truths.append(val_data_clean)
        results.append(clean)
        val_text.append(row['sentence'])
        label_result.append(answer +". "+ reason)

    print("Evaluation completed.")
    print("Saving results...")
    data_to_save = {
        "predictions": results,
        "ground_truths": ground_truths
    }

    rag_analysis = {
        "text": val_text,
        "llm_result": label_result
    }

    results_filename = "rag_evaluation_results"
    mer_analysis_filename = "rag_analysis"
    save_results_to_file(data_to_save, results_filename, 'rag', prompt_language)
    save_analysis_results(rag_analysis, mer_analysis_filename, 'rag', prompt_language)
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, default='gemini-2.5-flash')
    parser.add_argument('--prompt_language', type=str, default='english', help='Language for the prompt (english/indonesian/balinese)')
    args = parser.parse_args()

    keys = load_environment_variables()
    run_rag(keys, args.llm_name, args.prompt_language)
