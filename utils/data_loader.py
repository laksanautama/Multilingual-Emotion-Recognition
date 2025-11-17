import os
import pandas as pd
from .model_config import DATA_CONFIG, TRAINING_CONFIG
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import login
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer 



TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'crosslingual_ER',
                            DATA_CONFIG["TEST_DATA_DIR"]
                            )

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         'crosslingual_ER',
                        DATA_CONFIG["CACHE_DIR"]
                        )

def pos_weight(labels):
    """ Calculate positive weight for each label"""
    labels_np = np.array(labels)
    pos_weights = []
    num_labels = DATA_CONFIG["NUM_LABELS"]

    for i in range(num_labels):
        pos = labels_np[:, i].sum()
        neg = labels_np.shape[0] - pos
        pos_weight = (neg / pos) if pos > 0 else 1.0
        pos_weights.append(pos_weight)
    
    pos_weights = torch.tensor(pos_weights, dtype=torch.float)
    return pos_weights

def get_label_binarizer(dataset):
    """ Get MultiLabelBinarizer for labels """
    columns_name = DATA_CONFIG["LABELS"]
    mlb = MultiLabelBinarizer()
    mlb.fit([columns_name])

    emotion_string = dataset['emotions']
    if not isinstance(emotion_string, str):
        if isinstance(emotion_string, list):
            emotion_list = [str(e).strip().strip("'\"") for e in emotion_string if isinstance(e, str) and str(e).strip()]
        else:
            dataset['labels'] = np.zeros(len(columns_name), dtype=np.float32)
            return dataset
    else:
        emotion_list = [str(e).strip().strip("'\"") for e in emotion_string.strip("[]").split(',') if str(e).strip()]

    if emotion_list:
        multi_hot_vector = mlb.transform([emotion_list])[0]
    else:
        multi_hot_vector = np.zeros(len(columns_name), dtype=np.float32)
    
    dataset['labels'] = np.array(multi_hot_vector, dtype=np.float32)
    return dataset


def load_target_test_data(filename: str, cross_lingual: bool = False):
    """ Load balinese language test data 
    from CSV file."""

    file_path = os.path.join(TEST_DATA_DIR, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {filename} does not exist in the test data directory.")
    
    test_df = pd.read_csv(file_path)

    if cross_lingual:

        target_dataset = Dataset.from_pandas(test_df)
        print(f"Successfully loaded test dataset")
        if '__index_level_0__' in target_dataset.column_names:
            target_dataset = target_dataset.remove_columns(["__index_level_0__"])
        mhe_target_dataset = target_dataset.map(get_label_binarizer)
        # split = mhe_target_dataset.train_test_split(test_size=0.5, seed=TRAINING_CONFIG["SEED"])
        # val_data = split['train']
        # test_data = split['test']
        return mhe_target_dataset
        
    else:
        return test_df

def load_huggingface_dataset(dataset_name: str, dataset_lang: list, split: str, keys: dict):
    """ Load indonesia, java, and sunda language 
    dataset from Huggingface Hub."""
    hf_token = keys.get("HUGGINGFACE_TOKEN")
    print(f"Token HG: {hf_token}")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
    else:
        print("Warning: Hugging Face Token not provided. Will only be able to access public models/datasets.")
    print(f"dataset name: {dataset_name}")
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
        mhe_dataset = combined_dataset.map(get_label_binarizer)
        pos_weights_label = pos_weight(mhe_dataset['labels'])
        print(f"Combined dataset contains {len(combined_dataset)} samples from languages: {', '.join(dataset_lang)} with split: {split}")
        # SPLIT_CACHE_DIR = os.path.join(CACHE_DIR, split)
        # mhe_dataset.save_to_disk(SPLIT_CACHE_DIR)
        # print(f"Combined {split} dataset saved to cache directory: {SPLIT_CACHE_DIR}")
        return mhe_dataset, pos_weights_label