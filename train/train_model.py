import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from transformers import GPT2LMHeadModel
from torch.optim import AdamW
from tqdm import tqdm
from data.prepare_data import prepare_data

def train_model(num_epochs=3, learning_rate=5e-5):
    # Prepare data
    train_loader, val_loader, tokenizer = prepare_data()
    
    # Initialize model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1} - Average validation loss: {avg_val_loss:.4f}')
        
        # Save checkpoint
        checkpoint_path = f'models/checkpoint-epoch-{epoch + 1}'
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

if __name__ == "__main__":
    train_model() 