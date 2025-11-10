from .model_configs import DATA_CONFIG, MODEL_PATHS, TRAINING_CONFIG, MODEL_CHECKPOINTS, MODEL_SAVE
from .data_loader import load_target_test_data, load_huggingface_dataset
from .emotion_dataset import EmotionDataset
from sklearn.model_selection import train_test_split
from utils import check_lmmodel_exists, save_lmmodel, clear_gpu_memory, save_crosslingual_results
from .classifier import get_trainer
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

import torch

def load_lmmodel_config(lm_name: str):
    MODEL_NAME = MODEL_PATHS[lm_name]
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = DATA_CONFIG["NUM_LABELS"]
    config.problem_type = "multi_label_classification"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    
    return tokenizer, model

def inference_lmmodel(model_name: str, test_data):
    MAIN_DIR = (os.path.dirname(__file__))
    saved_model_dir = os.path.join(MAIN_DIR, MODEL_SAVE[model_name])
    tokenizer = AutoTokenizer.from_pretrained(saved_model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir)
    ##Continue from here to add inference logic

def run(keys: dict, lm_name: str):
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    """Placeholder function for running crosslingual ER tasks."""
    print("Running crosslingual tasks with provided API keys...")
    try:
        val_dataset, test_dataset = load_target_test_data(DATA_CONFIG["TARGET_TEST_FILENAME"], cross_lingual=True)
        # val_data, test_data = train_test_split(test_dataset, test_size=0.5, random_state=TRAINING_CONFIG["SEED"], shuffle=True)
        source_dataset_train, pw_label_train = load_huggingface_dataset(DATA_CONFIG["SOURCE_HF_DATASET"], DATA_CONFIG["DATASET_LANGUAGES"], 'train', keys)
        print("Datasets loaded successfully.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        print("Please ensure the test data file is present in the data/test_data directory.")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
   
    if check_lmmodel_exists(lm_name) == False:
        print(f"Model {lm_name} is not found. Proceeding with training...")
        tokenizer, model = load_lmmodel_config(lm_name)

        em_training_dataset = EmotionDataset(source_dataset_train, tokenizer, DATA_CONFIG["TRAIN_TEXT_COLUMN"], DATA_CONFIG["LABEL_COLUMN"], TRAINING_CONFIG["MAX_SEQ_LENGTH"])
        em_test_dataset = EmotionDataset(val_dataset, tokenizer, DATA_CONFIG["TEST_TEXT_COLUMN"], DATA_CONFIG["LABEL_COLUMN"], TRAINING_CONFIG["MAX_SEQ_LENGTH"])

        trainer = get_trainer(TRAINING_CONFIG['BATCH_SIZE'], TRAINING_CONFIG['EPOCHS'], TRAINING_CONFIG['LEARNING_RATE'], model, tokenizer, pw_label_train, em_training_dataset, em_test_dataset, MODEL_CHECKPOINTS[lm_name])
        # Train the model
        print(f"\n{'='*50}")
        print("Starting Training...")
        print(f"{'='*50}")
        trainer.train()
        print("Training completed.")
        dev_results = trainer.evaluate()
        model_result = {}
        results = {
                "eval_loss": dev_results["eval_loss"],
                "f1_macro": dev_results["eval_f1_macro"],
                "accuracy": dev_results["eval_accuracy"],
                "precision_macro": dev_results["eval_precision_macro"],
                "recall_macro": dev_results["eval_recall_macro"],
                "hamming_loss": dev_results["eval_hamming_loss"]
                }

        model_result[lm_name] = results

        print(f"f1_score: {results['f1_macro']}")
        print(f"accuracy: {results['accuracy']}")
        save_crosslingual_results(model_result)
        save_lmmodel(model, tokenizer, lm_name)
        clear_gpu_memory(model=model, tokenizer=tokenizer)
        del model
        del tokenizer
        # Save the trained model.....
        # Add model evaluation logic here
    else:
        pass



    
    
    

    
    
