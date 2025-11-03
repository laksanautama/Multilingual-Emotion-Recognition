from utils import check_faiss_exists, llm_dataset_preparation, create_val_examples, save_results_to_file, save_analysis_results
from llm_evaluation.llm_code.llm_utility.llm_utility import llm_selection, select_language_config, strip_result_content
from crosslingual_ER.scripts.model_configs import DATA_CONFIG
from llm_evaluation.llm_code.llm_utility.llm_config import DIRECTORY_PATH, PROMPT_CONFIG
from .rag_pipeline import store_vectors_database, rag_retriever, rag_classifier
from langchain_core.prompts import PromptTemplate
import json
import os

def run_rag(keys: dict, llm_model_name: str, prompt_language: str):
    """Placeholder function for running RAG pipeline evaluation tasks."""
    print("-" * 20)
    print("Balinese Text Emotion Recognition using RAG Adaptation Method")
    print("-" * 20)

    MAIN_DIR = (os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    results_dir = os.path.join(MAIN_DIR, DIRECTORY_PATH["RAG_RESULTS_DIR"])

    train_data, val_data = llm_dataset_preparation(DATA_CONFIG["TARGET_TEST_FILENAME"], DATA_CONFIG["TRAIN_SPLIT_SIZE"])
    val_samples = create_val_examples(val_data, DATA_CONFIG["NUM_VAL_SAMPLES"])

    if check_faiss_exists('index') is False:
        store_vectors_database(train_data, keys)
    
    config = select_language_config(prompt_language)
    llm = llm_selection(llm_model_name, keys)
    retriever = rag_retriever(llm_model_name, keys, DATA_CONFIG["RAG_TOP_K"])
    
    classification_prompt = PromptTemplate(
        template=config['clf_template'],
        input_variables=["context", "query", "labels"]
    )
    chain = classification_prompt | llm

    results = []
    ground_truths = []
    val_text = []
    label_result = []

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

    # results_filepath = f"{results_dir}/rag_evaluation_results.json"
    # with open(results_filepath, "w") as f:
    #     json.dump(data_to_save, f, indent=4)
    # print(f"Results saved to {results_filepath}.")
    # Continue from here to implement the evaluation using rag_model on val_data