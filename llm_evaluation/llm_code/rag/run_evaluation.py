from utils import check_faiss_exists, llm_dataset_preparation
from crosslingual_ER.scripts.model_configs import DATA_CONFIG
from .rag_pipeline import store_vectors_database

def run_rag(keys: dict, llm_model_name: str):
    """Placeholder function for running RAG pipeline evaluation tasks."""
    print("-" * 20)
    print("Balinese Text Emotion Recognition using RAG Pipeline Evaluation")
    print("-" * 20)

    train_data, val_data = llm_dataset_preparation(DATA_CONFIG["TARGET_TEST_FILENAME"], DATA_CONFIG["TRAIN_SPLIT_SIZE"])
    if check_faiss_exists('index') is False:
        store_vectors_database(train_data, keys)