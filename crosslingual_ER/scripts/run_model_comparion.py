from .model_configs import DATA_CONFIG, MODEL_PATHS, TRAINING_CONFIG
from .data_loader import load_target_test_data, load_huggingface_dataset
import torch

def run(keys: dict):
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    """Placeholder function for running crosslingual ER tasks."""
    print("Running crosslingual tasks with provided API keys...")
    try:
        test_dataset = load_target_test_data(DATA_CONFIG["TARGET_TEST_FILENAME"], cross_lingual=True)
        source_dataset_train, pw_label_train = load_huggingface_dataset(DATA_CONFIG["SOURCE_HF_DATASET"], DATA_CONFIG["DATASET_LANGUAGES"], 'train', keys)
        print("Datasets loaded successfully.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        print("Please ensure the test data file is present in the data/test_data directory.")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
