from crosslingual_ER.scripts.data_loader import load_target_test_data
from sklearn.model_selection import train_test_split
from crosslingual_ER.scripts.model_configs import TRAINING_CONFIG, DATA_CONFIG
from llm_evaluation.llm_code.llm_utility.llm_config import DIRECTORY_PATH
import pandas as pd
import math
import json
import os
import logging

def create_train_examples(train_dataset, emotion, total_samples: int):
    """ create num_examples * 2 of samples from train dataset"""
    try:
        if total_samples > len(train_dataset):
            raise ValueError("Requested more samples than available in the training dataset.")
        
        pos_num_examples = math.ceil(total_samples // 2)
        neg_num_examples = total_samples - pos_num_examples

        n_samples_1 = min(pos_num_examples, len(train_dataset[train_dataset[emotion] == 1]))
        n_samples_0 = min(neg_num_examples, len(train_dataset[train_dataset[emotion] == 0]))
        
        example = []
        samples_1 = train_dataset[train_dataset[emotion] == 1].sample(n_samples_1, random_state=TRAINING_CONFIG["SEED"])
        samples_0 = train_dataset[train_dataset[emotion] == 0].sample(n_samples_0, random_state=TRAINING_CONFIG["SEED"])
        samples_full = pd.concat([samples_1, samples_0])

        for i in range(len(samples_full)):
            answer = 'yes' if samples_full[emotion].iloc[i] == 1 else 'no'
            example.append({'input': samples_full['sentence'].iloc[i], 'answer': answer})
    
        return example
    
    except ValueError as e:
        print(f"Error in creating training examples: {e}")
        raise

def create_val_examples(val_dataset, val_samples: int):
    """ Create validation examples from val dataset. """
    num_samples = min(val_samples, len(val_dataset))

    def sample_group(group):
        return group.sample(min(len(group), max(1, int(num_samples * len(group) / len(val_dataset)))), random_state=TRAINING_CONFIG["SEED"])
    
    val_data_samples = val_dataset.groupby('emotions').apply(sample_group).sample(min(num_samples, len(val_dataset)), random_state=TRAINING_CONFIG["SEED"])
    return val_data_samples.reset_index(drop=True)

def llm_dataset_preparation(filename: str, split_size: float):
    """Placeholder function for preparing datasets for few-shot adaptation."""
    dataset = load_target_test_data(filename, cross_lingual=False)
    # dataset will be split into training and validation sets based on split_size
    train_data, val_data = train_test_split(dataset, test_size=split_size, random_state=TRAINING_CONFIG["SEED"])
    
    return train_data.reset_index(drop=True), val_data.reset_index(drop=True)

def save_results_to_file(results: dict, filename: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    """ Save results dictionary to a text file. """
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(MAIN_DIR, DIRECTORY_PATH["FEW_SHOT_RESULTS_DIR"])
    filename_path = f"{full_path}/{filename}.json"
    os.makedirs(full_path, exist_ok=True)
    with open(filename_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {filename_path}")

