from .data_loader  import load_target_test_data
from sklearn.model_selection import train_test_split
from .model_config import TRAINING_CONFIG, DATA_CONFIG, MODEL_SAVE, MODEL_CHECKPOINTS
from .llm_configs import DIRECTORY_PATH
from .llm_utility import translate_label, translate_answer, select_language_config
import pandas as pd
import math
import json
import os
import logging

def proportional_language_sample(df_sub, target_n):
    """
    Sample rows from df_sub proportional to its language distribution.
    """
    # Compute distribution
    lang_dist = df_sub["language"].value_counts(normalize=True)

    # Compute how many samples per language
    samples_per_lang = (lang_dist * target_n).round().astype(int)

    # Fix rounding issues (may overshoot or undershoot)
    diff = target_n - samples_per_lang.sum()

    if diff > 0:
        # Add missing samples to languages with largest residual probability
        remainder_order = (lang_dist * target_n - samples_per_lang).sort_values(ascending=False)
        for lang in remainder_order.index[:diff]:
            samples_per_lang[lang] += 1
    elif diff < 0:
        # Remove extras from largest allocations
        remainder_order = samples_per_lang.sort_values(ascending=False)
        for lang in remainder_order.index[:abs(diff)]:
            samples_per_lang[lang] -= 1

    # Perform group-wise sampling
    sampled_parts = []
    for lang, k in samples_per_lang.items():
        lang_df = df_sub[df_sub["language"] == lang]

        # Adjust k if language doesn't have enough samples
        k = min(k, len(lang_df))
        if k > 0:
            sampled_parts.append(lang_df.sample(k, random_state=TRAINING_CONFIG["SEED"]))

    return pd.concat(sampled_parts) if sampled_parts else pd.DataFrame()


def create_train_examples(train_dataset, emotion, total_samples: int, prompt_language: str):
    """
    Create positive/negative samples from a HuggingFace dataset,
    with proportional sampling based on the 'languages' column.
    """
    try:
        # Convert HF dataset → pandas for flexible grouping/sampling
        df = train_dataset.to_pandas()

        if total_samples > len(df):
            raise ValueError("Requested more samples than available in the training dataset.")

        # Split into positive and negative
        df_pos = df[df[emotion] == 1]
        df_neg = df[df[emotion] == 0]

        # Compute desired sample counts
        pos_target = math.ceil(total_samples / 2)
        neg_target = total_samples - pos_target

        pos_n = min(pos_target, len(df_pos))
        neg_n = min(neg_target, len(df_neg))

         # Apply proportional sampling
        sampled_pos = proportional_language_sample(df_pos, pos_n)
        sampled_neg = proportional_language_sample(df_neg, neg_n)

        # Combine
        samples_full = pd.concat([sampled_pos, sampled_neg], ignore_index=True)

        # Re-shuffle
        samples_full = samples_full.sample(len(samples_full), random_state=TRAINING_CONFIG["SEED"])

        # Load config
        config = select_language_config(prompt_language)

        # Build final example list
        example = []
        for idx, row in samples_full.iterrows():
            en_answer = 'yes' if row[emotion] == 1 else 'no'
            conclusion = ""
            if en_answer == 'yes':
                conclusion = f"The emotion: {emotion} is present in this text"
            else:
                conclusion = f"The emotion: {emotion} is absent in this text"

            answer = translate_answer(en_answer, prompt_language)

            example.append({
                config['input']: row['text'],
                config['answer']: answer,
                "conclusion": conclusion
                # "language": row["language"]
            })

        return example
    
    except ValueError as e:
        print(f"Error in creating training examples: {e}")
        raise
# def create_train_examples(train_dataset, emotion, total_samples: int, prompt_language: str):
    
    
#     """ create num_examples * 2 of samples from train dataset"""
#     try:
#         if total_samples > len(train_dataset):
#             raise ValueError("Requested more samples than available in the training dataset.")
        
#         pos_num_examples = math.ceil(total_samples // 2)
#         neg_num_examples = total_samples - pos_num_examples

#         n_samples_1 = min(pos_num_examples, len(train_dataset[train_dataset[emotion] == 1]))
#         n_samples_0 = min(neg_num_examples, len(train_dataset[train_dataset[emotion] == 0]))
        
#         example = []
#         samples_1 = train_dataset[train_dataset[emotion] == 1].sample(n_samples_1, random_state=TRAINING_CONFIG["SEED"])
#         samples_0 = train_dataset[train_dataset[emotion] == 0].sample(n_samples_0, random_state=TRAINING_CONFIG["SEED"])
#         samples_full = pd.concat([samples_1, samples_0])
#         # _, _, _, _, input_key, answer_key = select_language_config(prompt_language)
#         config = select_language_config(prompt_language)

#         for i in range(len(samples_full)):
#             en_answer = 'yes' if samples_full[emotion].iloc[i] == 1 else 'no'
#             answer = translate_answer(en_answer, prompt_language)
#             example.append({config['input']: samples_full['sentence'].iloc[i], config['answer']: answer})

#         return example
    
#     except ValueError as e:
#         print(f"Error in creating training examples: {e}")
#         raise


def create_val_examples(val_dataset, val_samples: int):
    """ Create validation examples from val dataset. """
    num_samples = min(val_samples, len(val_dataset))
    # print(val_dataset)
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

def save_lmmodel(model, tokenizer, model_name: str):
    """ Save the fine-tuned LM model and tokenizer. """
    path = MODEL_SAVE[model_name]
    
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(MAIN_DIR, 'crossligual_ER', path)
    os.makedirs(full_path, exist_ok=True)
    model.save_pretrained(full_path)
    tokenizer.save_pretrained(full_path)
    print(f"Model and tokenizer saved to {full_path}")

def save_results_to_file(results: dict, filename: str, task: str, prompt_language: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    """ Save results dictionary to a text file. """
    if task == "few_shot":
        dir = DIRECTORY_PATH["FEW_SHOT_RESULTS_DIR"]
    elif task == "rag":
        dir = DIRECTORY_PATH["RAG_RESULTS_DIR"]
    elif task == "zero_shot":
        dir = DIRECTORY_PATH["ZERO_SHOT_RESULTS_DIR"]
    elif task == "multiagents":
        dir = DIRECTORY_PATH["MULTIAGENTS_RESULT_DIR"]

    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(MAIN_DIR, dir, prompt_language)
    filename_path = f"{full_path}/{filename}.json"
    os.makedirs(full_path, exist_ok=True)
    with open(filename_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results {task} saved to {filename_path}")

def save_analysis_results(results: dict, filename: str, task: str, prompt_language: str):
    """ Save results dictionary to a text file. """
    if task == "few_shot":
        dir = DIRECTORY_PATH["FEW_SHOT_RESULTS_DIR"]
    elif task == "rag":
        dir = DIRECTORY_PATH["RAG_RESULTS_DIR"]
    elif task == "zero_shot":
        dir = DIRECTORY_PATH["ZERO_SHOT_RESULTS_DIR"]
    elif task == "multiagents":
        dir = DIRECTORY_PATH["MULTIAGENTS_RESULT_DIR"]

    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(MAIN_DIR, dir, prompt_language)
    filename_path = f"{full_path}/{filename}.json"

    if os.path.exists(filename_path) and os.path.getsize(filename_path) > 0:
        with open(filename_path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: JSON file at {filename_path} is corrupted or empty. Starting new dictionary.")
                existing_data = {}
    else:
        existing_data = {}
    
    existing_data.update(results)

    with open(filename_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
        
    print(f"Data successfully appended and saved to {filename_path}")
    
def save_crosslingual_results(results: dict):
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(MAIN_DIR, "results", "crosslingual_ER")
    filename_path = os.path.join(full_path, "crosslingual_results.json")

    try:
        
        os.makedirs(full_path, exist_ok=True)

        if os.path.exists(filename_path) and os.path.getsize(filename_path) > 0:
            try:
                with open(filename_path, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
        else:
            existing_data = {}
        
        existing_data.update(results)

        with open(filename_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        print(f"Crosslingual results successfully saved to {filename_path}")
    except Exception as e:
        print(f"Error saving crosslingual results: {e}")

def save_lora_results(results: list):
    """ Save LoRA results dictionary to a text file. """

    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(MAIN_DIR, "results", "lora")
    filename_path = os.path.join(full_path, "lora_results.json")
    os.makedirs(full_path, exist_ok=True)
    with open(filename_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"LoRA Results saved to {filename_path}")

def load_json_file(filename: str, task: str, model: str = None):
    """ Load and return the contents of a JSON file. """
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    if task == "few_shot":
        dir = DIRECTORY_PATH["FEW_SHOT_RESULTS_DIR"]
    elif task == "rag":
        dir = DIRECTORY_PATH["RAG_RESULTS_DIR"]
    elif task == "zero_shot":
        dir = DIRECTORY_PATH["ZERO_SHOT_RESULTS_DIR"]
    elif task == "multiagents":
        dir = DIRECTORY_PATH["MULTIAGENTS_RESULT_DIR"]

    full_path = os.path.join(MAIN_DIR, dir, model)
    filepath = f"{full_path}/{filename}.json"

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            print("Successfully loaded JSON file:")
            return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {filepath}")
        return None

def check_faiss_exists(filename: str) -> bool:
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(MAIN_DIR, DATA_CONFIG["FAISS_INDEX_DIR"])
    filepath = f"{full_path}/{filename}.faiss"
    return os.path.exists(filepath)


def check_lmmodel_exists(model_name: str) -> bool:
    
    path = MODEL_SAVE[model_name]
    
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(MAIN_DIR, path)
    model_path = f"{full_path}"
    return os.path.exists(f"{model_path}/tokenizer.json") and os.path.exists(f"{model_path}/model.safetensors")

def check_tokenizer_and_model_exists(saved_model: str, tokenizer: str) -> bool:
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(MAIN_DIR, DIRECTORY_PATH["LORA_CHECKPOINTS_DIR"])
    model_path = f"{full_path}/{saved_model}.safetensors"
    tokenizer_path = f"{full_path}/{tokenizer}.json"
    return os.path.exists(model_path) and os.path.exists(tokenizer_path)

def get_folder_name(dir_path: str):
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(MAIN_DIR, dir_path)
    folder_name = []
    all_entries = os.listdir(full_path)
    for entry in all_entries:
        entry_path = os.path.join(full_path, entry)
        if os.path.isdir(entry_path):
            folder_name.append(entry)
    return folder_name

