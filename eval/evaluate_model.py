from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import os

def load_best_model():
    # Find the latest checkpoint
    checkpoints = [d for d in os.listdir('models') if d.startswith('checkpoint-epoch-')]
    if not checkpoints:
        raise ValueError("No checkpoints found in models directory")
    
    latest_checkpoint = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join('models', latest_checkpoint)
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=100):
    # Create text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Generate response
    response = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    return response[0]['generated_text']

def evaluate_model():
    # Load the best model
    model, tokenizer = load_best_model()
    
    # Test prompts
    test_prompts = [
        "Alex: Do you like movies?\nBob:",
        "Alex: What's your favorite programming language?\nBob:",
        "Alex: How are you today?\nBob:"
    ]
    
    print("\nGenerating responses for test prompts:")
    print("-" * 50)
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_response(prompt, model, tokenizer)
        print(f"Generated response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_model() 