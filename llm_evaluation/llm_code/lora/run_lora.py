from llm_evaluation.llm_code.llm_utility.llm_config import DIRECTORY_PATH
from crosslingual_ER.scripts.model_configs import MODEL_PATHS, DATA_CONFIG
from utils import check_tokenizer_and_model_exists, llm_dataset_preparation, create_val_examples
from train_lora import train_lora
from evaluate_lora import inference_lora

def run_lora(keys: dict, llm_model_name: str, prompt_language: str):
    
    train_data, val_data = llm_dataset_preparation(DATA_CONFIG["TARGET_TEST_FILENAME"], DATA_CONFIG["TRAIN_SPLIT_SIZE"])
    val_samples = create_val_examples(val_data, DATA_CONFIG["NUM_VAL_SAMPLES"])
    if check_tokenizer_and_model_exists('adapter_model', 'tokenizer') is False:
        train_lora(train_data)
    else:
        inference_lora(val_samples)