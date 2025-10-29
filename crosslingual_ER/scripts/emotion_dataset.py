import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
  def __init__(self, hf_dataset, tokenizer, text_column, label_column='labels', max_length=256):
    self.dataset = hf_dataset
    self.tokenizer = tokenizer
    self.text_column = text_column
    self.label_column = label_column
    self.max_length = max_length
    
  def __len__(self):
    return len(self.dataset)
    
  def __getitem__(self, idx):
    item = self.dataset[idx]
    text = str(item[self.text_column])
    labels = item[self.label_column]
        
  # Tokenize text
    encoding = self.tokenizer(
          text,
          max_length=self.max_length,
          padding='max_length',
          truncation=True,
          return_tensors='pt'
          )
        
    
    return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }