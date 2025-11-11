from crosslingual_ER.scripts.model_configs import MODEL_PATHS
from .lora_config import load_lora_config
import torch
from utils import save_lora_results
from llm_evaluation.llm_code.llm_utility.llm_config import PROMPT_CONFIG, DIRECTORY_PATH

import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

def inference_lora(val_dataset):
    print("-" * 20)
    print("Starting LoRA Inference...")
    print("-" * 20)
    PARENT_DIR = (os.path.dirname(os.path.dirname(__file__)))
    MODEL_DIR = os.path.join(PARENT_DIR, DIRECTORY_PATH["LORA_MODEL_DIR"])

    bnb_config, _ = load_lora_config()

    base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS["meta_llama"],
            quantization_config=bnb_config,
            device_map="auto",
            )
    
    def predict(text, tokenizer):

        prompt = (
                f"{SYSTEM_INSTRUCT}\n"
                f"Text: {text}\n"
                f"Answer:"
            )
        
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            output_ids = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0.3
                    )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        emotions_str = decoded.split("###")[0].strip()
        emotions_list = [e.strip().replace('.', '') for e in emotions_str.split(',') if e.strip()]
        return emotions_list
    
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    predictions = []
    for _, row in val_dataset.iterrows():
        prediction = predict(row['sentence'], tokenizer)
        predictions.append({
            'sentence': row['sentence'],
            'true_emotions': row['emotions'],
            'predicted_emotions': prediction
        })

        print(f"Input Text: {row['sentence']}")
        print(f"Predicted Emotions: {prediction}")
        print("-" * 20)
    
    save_lora_results(predictions)



def run_cli():
    pass

if __name__ == "__main__":
    run_cli()