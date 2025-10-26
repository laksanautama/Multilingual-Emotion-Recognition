from .model_configs import DATA_CONFIG, MODEL_PATHS, TRAINING_CONFIG
from .data_loader import load_target_test_data, load_huggingface_dataset

def run(api_keys: dict):
    """Placeholder function for running crosslingual ER tasks."""
    print("Running crosslingual tasks with provided API keys...")
    try:
        test_dataset = load_target_test_data(DATA_CONFIG["TARGET_TEST_FILENAME"])
        source_dataset = load_huggingface_dataset(DATA_CONFIG["SOURCE_HF_DATASET"], DATA_CONFIG["DATASET_LANGUAGES"])
        print("Datasets loaded successfully.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        print("Please ensure the test data file is present in the data/test_data directory.")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")