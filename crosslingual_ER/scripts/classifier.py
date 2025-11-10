
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import os
import torch

class CustomTrainer(Trainer):
  def __init__(self, pos_weights, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.pos_weights = pos_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
      labels = inputs.pop("labels")
      outputs = model(**inputs)
      logits = outputs.logits
      loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
      loss = loss_fct(logits, labels)
      return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """Compute metrics for multi-label evaluation"""
    predictions, labels = eval_pred
    # Apply sigmoid to the logits to get probabilities
    probabilities = 1 / (1 + np.exp(-predictions))
    # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
    binary_predictions = (probabilities > 0.5).astype(int)

    # Macro-F1 across all emotions
    f1_macro = f1_score(labels, binary_predictions, average="macro", zero_division=0)
    return {"f1_macro": f1_macro}


def get_trainer(batch_size, epochs, lr, model, tokenizer, pos_weights, train_dataset, eval_dataset, lm_dir):

  MAIN_DIR = (os.path.dirname(__file__))


  training_args = TrainingArguments(
      output_dir=os.path.join(MAIN_DIR, lm_dir),
      num_train_epochs=epochs,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      learning_rate=lr,
      logging_dir='./logs',
      logging_steps=100,
      eval_strategy='epoch',
      save_strategy='no',
      report_to='none',
    )
  
  trainer = CustomTrainer(
    pos_weights = pos_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # U
    eval_dataset=eval_dataset,    
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
  )
  return trainer