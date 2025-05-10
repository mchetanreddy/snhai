# LLM Dialogue Fine-tuning Pipeline

This project implements a pipeline for fine-tuning Large Language Models (LLMs) on dialogue datasets. The implementation focuses on creating a reproducible and efficient training process with proper evaluation metrics.

## Project Structure

```
├── data_prep.py      # Data loading and preprocessing
├── train.py         # Model training and fine-tuning
├── evaluate.py      # Model evaluation and generation
├── requirements.txt # Project dependencies
└── README.md       # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preprocessing

The preprocessing pipeline:
- Loads JSON dialogue data
- Formats dialogues with special tokens for speaker separation
- Tokenizes using GPT-2 tokenizer with custom special tokens
- Splits data into training (80%) and validation (20%) sets
- Creates PyTorch datasets for efficient training

## Model Architecture

Initial implementation uses GPT-2 Medium (345M parameters) as the base model, which provides a good balance between:
- Training speed and resource requirements
- Model capacity for dialogue generation
- Quality of generated outputs

## Training Configuration

- Optimizer: AdamW with learning rate 5e-5
- Batch size: 8 (adjustable based on GPU memory)
- Max sequence length: 512 tokens
- Training epochs: 3
- Gradient accumulation steps: 4
- Mixed precision training enabled

## Evaluation

The model is evaluated on:
- Validation perplexity
- Sample dialogue generation quality
- Context retention
- Response coherence

## Usage

1. Prepare the data:
```bash
python data_prep.py
```

2. Train the model:
```bash
python train.py
```

3. Evaluate and generate samples:
```bash
python evaluate.py
```

## Results

Training metrics and sample generations will be logged to Weights & Biases for easy visualization and tracking. 