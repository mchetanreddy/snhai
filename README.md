# Dialogue-based LLM Training Pipeline

This project implements a training pipeline for fine-tuning a GPT-2 model on dialogue data. The pipeline includes data preparation, model training, and evaluation components.

## Project Structure

```
.
├── data/
│   └── prepare_data.py    # Data preprocessing and dataset creation
├── train/
│   └── train_model.py     # Model training script
├── eval/
│   └── evaluate_model.py  # Model evaluation and text generation
├── models/                # Directory for saved model checkpoints
├── dialogues.json         # Training data
└── requirements.txt       # Project dependencies
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
```bash
python data/prepare_data.py
```

2. Model Training:

**Option 1 (Recommended):**
Run from the project root using the `-m` flag:
```bash
python -m train.train_model
```

**Option 2:**
Run the script directly (the script now handles the import path automatically):
```bash
python train/train_model.py
```

3. Model Evaluation:
```bash
python eval/evaluate_model.py
```

## Model Architecture

- Base Model: GPT-2
- Training Parameters:
  - Learning rate: 5e-5
  - Batch size: 4
  - Number of epochs: 3
  - Max sequence length: 128

## Dataset Format

The training data is stored in `dialogues.json` with the following format:
```json
[
    {
        "Alex": "Hello, how are you?",
        "Bob": "I'm doing well, thank you. How about you?",
        "Alex": "I'm good too."
    },
    ...
]
```

## Model Checkpoints

Model checkpoints are saved in the `models/` directory after each epoch. The latest checkpoint is automatically used for evaluation.

## Evaluation

The evaluation script generates responses for sample prompts and demonstrates the model's ability to engage in dialogue. Sample prompts include:
- "Alex: Do you like movies?\nBob:"
- "Alex: What's your favorite programming language?\nBob:"
- "Alex: How are you today?\nBob:" 