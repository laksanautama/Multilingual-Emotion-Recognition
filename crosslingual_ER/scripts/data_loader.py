import os
import pandas as pd
from .model_configs import DATA_CONFIG
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import login


TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    DATA_CONFIG["TEST_DATA_DIR"]
)
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    DATA_CONFIG["CACHE_DIR"]
)

def load_target_test_data(filename: str):
    """ Load balinese language test data 
    from CSV file."""

    file_path = os.path.join(TEST_DATA_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {filename} does not exist in the test data directory.")
    
    print(f"Loading test data from: {file_path}")
    try:
        test_df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(test_df)} samples from {filename}.")
        try:
            target_dataset = Dataset.from_pandas(test_df)
        except Exception as e:
            print(f"Error converting test set from DataFrame to Dataset: {e}")
            raise

        if '__index_level_0__' in target_dataset.column_names:
            target_dataset = target_dataset.remove_columns(["__index_level_0__"])
        return target_dataset
    
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        raise

def load_huggingface_dataset(dataset_name: str, dataset_lang: list, split: str, keys: dict):
    """ Load indonesia, java, and sunda language 
    dataset from Huggingface Hub."""
    hf_token = keys.get("HUGGINGFACE_TOKEN")
    print(f"Token HG: {hf_token}")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
    else:
        print("Warning: Hugging Face Token not provided. Will only be able to access public models/datasets.")
    
    loaded_datasets = []
    for lang in dataset_lang:
        loaded = False
        try:
            dataset = load_dataset(dataset_name, lang, split=split)
            print(f"Successfully loaded {lang} samples with split: {split}")
            loaded_datasets.append(dataset)
            loaded = True
        except ValueError as e:
            if "Split 'train' not found" in str(e) or "Should be one of ['dev', 'test']" in str(e):
                try:
                    dataset = load_dataset(dataset_name, lang, split='dev')
                    print(f"Successfully loaded {lang} samples with split: 'dev'")
                    loaded_datasets.append(dataset)
                    loaded = True
                except Exception as e:
                    print(f"Error loading 'train' or 'dev' split for {lang}: {e}")
            else:
                print(f"General error loading dataset {lang}: (non-split error): {e}")
        
        except Exception as e:
            print(f"Error loading dataset {lang}: {e}")
    
    if loaded_datasets:
        combined_dataset = concatenate_datasets(loaded_datasets)
        print(f"Combined dataset contains {len(combined_dataset)} samples from languages: {', '.join(dataset_lang)} with split: {split}")
        SPLIT_CACHE_DIR = os.path.join(CACHE_DIR, split)
        combined_dataset.save_to_disk(SPLIT_CACHE_DIR)
        print(f"Combined {split} dataset saved to cache directory: {SPLIT_CACHE_DIR}")
        return combined_dataset