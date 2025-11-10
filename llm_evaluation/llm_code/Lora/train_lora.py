from crosslingual_ER.scripts.model_configs import MODEL_PATHS
from utils import clear_gpu_memory
import pandas as pd
import torch
import os
from datasets import Dataset

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
from llm_evaluation.llm_code.llm_utility.llm_config import PROMPT_CONFIG, DIRECTORY_PATH

def train_lora(train_data):

    PARENT_DIR = (os.path.dirname(os.path.dirname(__file__)))

    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                )
    
    lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "v_proj"]
                )
    
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS["meta_llama"],
            quantization_config=bnb_config,
            device_map="auto",
            )

    def format_labels(label: str):
        str_emotion = label.strip("[]'").replace("'", "")
        emotion = 'no emotion' if str_emotion == '' else str_emotion
        emotion_label = emotion.split(", ")
        return emotion_label
    
    def make_prompt(example):
        prompt = (
        f"{PROMPT_CONFIG['SYSTEM_INSTRUCT']}\n"
        f"Text: {example['sentence']}\n"
        f"Answer:"
        )
        target = example["emotions_label"]
        return {
            "input_text": prompt,
            "target_text": target
        }
    
    def tokenize(batch, tokenizer):
        target_text = ", ".join(batch["target_text"])
        full_text = batch["input_text"] + " " + target_text
        out = tokenizer(
                        full_text,
                        truncation=True,
                        max_length=512,
                        padding="max_length"
                    )
        # create labels: everything except target should be -100 (ignored)
        input_ids = out["input_ids"]
        labels = input_ids.copy()

        prompt_len = len(tokenizer(batch["input_text"])["input_ids"])
        for i in range(prompt_len - 1):   # ignore prompt tokens
            labels[i] = -100

        out["labels"] = labels
        return out
    
    train_data['emotions_label'] = train_data['emotions'].apply(format_labels)
    dataset = Dataset.from_pandas(train_data)
    prompt_dataset = dataset.map(make_prompt)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["meta_llama"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = prompt_dataset.map(lambda x: tokenize(x, tokenizer), batched=False, remove_columns=prompt_dataset.column_names)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
            output_dir=f"{PARENT_DIR}/{DIRECTORY_PATH['LORA_CHECKPOINTS_DIR']}",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            logging_steps=20,
            num_train_epochs=3,
            learning_rate=2e-4,
            save_strategy="epoch",
            fp16=True,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            report_to="none"
        )
    
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
            )
        )
    
    trainer.train()
    print("Training completed.")
    print("Saving the LoRA model...")
    trainer.save_model(f"{PARENT_DIR}/{DIRECTORY_PATH['LORA_MODEL_DIR']}")
    tokenizer.save_pretrained(f"{PARENT_DIR}/{DIRECTORY_PATH['LORA_MODEL_DIR']}")
    print(f"LoRA model saved to {PARENT_DIR}/{DIRECTORY_PATH['LORA_MODEL_DIR']}.")
    clear_gpu_memory(model=model, tokenizer=tokenizer)
    del model
    del tokenizer

    
    

    



    