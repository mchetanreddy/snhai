import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np

class DialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=128):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        
        # Convert dialogue to text format
        text = ""
        for speaker, message in dialogue.items():
            text += f"{speaker}: {message}\n"
        
        # Tokenize the text
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }

def prepare_data():
    # Load the dialogues
    with open('dialogues.json', 'r') as f:
        dialogues = json.load(f)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = DialogueDataset(dialogues, tokenizer)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    train_loader, val_loader, tokenizer = prepare_data()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Print a sample batch
    sample_batch = next(iter(train_loader))
    print("\nSample batch shape:")
    print(f"Input IDs shape: {sample_batch['input_ids'].shape}")
    print(f"Attention mask shape: {sample_batch['attention_mask'].shape}") 