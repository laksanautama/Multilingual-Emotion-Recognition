from crosslingual_ER.scripts.model_configs import MODEL_PATHS
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

def inference_lora(val_dataset, bnb_config):

    PARENT_DIR = (os.path.dirname(os.path.dirname(__file__)))
    MODEL_DIR = os.path.join(PARENT_DIR, DIRECTORY_PATH["LORA_MODEL_DIR"])

    base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS["meta_llama"],
            quantization_config=bnb_config,
            device_map="auto",
            )
    
    def predict(text):
        prompt = (
                    f"{SYSTEM_INSTRUCT}\n"
                    f"Text: {text}\n"
                    f"Answer:"
                )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                                    **inputs,
                                    max_new_tokens=30,
                                    temperature=0.3
                        )
            
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded.split("Answer:")[-1].strip()
    
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    for _, row in val_dataset.iterrows():
        prediction = predict(row['sentence'])
        print(f"Input Text: {row['sentence']}")
        print(f"Predicted Emotions: {prediction}")
        print("-" * 20)



def run_cli():
    pass

if __name__ == "__main__":
    run_cli()