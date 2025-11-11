from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType

def load_lora_config():
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
    
    return bnb_config, lora_config