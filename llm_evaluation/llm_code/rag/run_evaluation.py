from utils import check_faiss_exists, llm_dataset_preparation
from llm_utility.llm_utility import llm_selection
from crosslingual_ER.scripts.model_configs import DATA_CONFIG
from .rag_pipeline import store_vectors_database, rag_retriever, rag_classifier
from langchain_core.prompts import PromptTemplate
from llm_utility.llm_utility import classify_text
import json
import os

def run_rag(keys: dict, llm_model_name: str):
    """Placeholder function for running RAG pipeline evaluation tasks."""
    print("-" * 20)
    print("Balinese Text Emotion Recognition using RAG Adaptation Method")
    print("-" * 20)

    MAIN_DIR = (os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    results_dir = os.path.join(MAIN_DIR, DATA_CONFIG["RAG_RESULTS_DIR"])

    train_data, val_data = llm_dataset_preparation(DATA_CONFIG["TARGET_TEST_FILENAME"], DATA_CONFIG["TRAIN_SPLIT_SIZE"])
    
    if check_faiss_exists('index') is False:
        store_vectors_database(train_data, keys)
    
    llm = llm_selection(llm_model_name, keys)
    retriever = rag_retriever(llm_model_name, keys, DATA_CONFIG["RAG_TOP_K"])
    classification_prompt = PromptTemplate(
        template=DATA_CONFIG["RAG_CLASSIFICATION_TEMPLATE"],
        input_variables=["context", "query", "labels"]
    )
    chain = classification_prompt | llm

    results = []
    ground_truths = []
    for _, row in val_data.iterrows():
        print(row['sentence'])
        print("-" * 15)
        label = rag_classifier(
            query = row['sentence'],
            retriever = retriever,
            chain = chain,
            labels = DATA_CONFIG["LABELS"]
          )
        text = label.content.replace("Answer:", "").strip()
        val_data_emotions = row['emotions'].strip("[]").replace("'", "")
        emotion = 'no emotion' if val_data_emotions == '' else val_data_emotions
        clean = text.split(", ")
        val_data_clean = emotion.split(", ")

        ground_truths.append(val_data_clean)
        results.append(clean)
    print("Evaluation completed.")
    print("Saving results...")
    data_to_save = {
        "predictions": results,
        "ground_truths": ground_truths
    }
    results_filepath = f"{results_dir}/rag_evaluation_results.pkl"
    with open(results_filepath, "w") as f:
        json.dump(data_to_save, f, indent=4)
    print(f"Results saved to {results_filepath}.")
    # Continue from here to implement the evaluation using rag_model on val_data