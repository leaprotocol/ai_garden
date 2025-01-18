---
title: TinyGPT Training Project
public: true
---

# 🌱 TinyGPT Training Project

A small-scale experiment in training a tokenizer and language model from scratch using the TinyStories dataset.

## 🎯 Project Goals

- Train a custom tokenizer from scratch
- Understand token-level operations in language models
- Visualize how text is broken down into tokens
- Experiment with different generation parameters

## 🛠️ Components

### Training Script (`main.py`)
- Loads and processes the TinyStories dataset
- Trains a custom tokenizer with vocabulary size of 8192
- Implements checkpoint saving and resumption
- Handles interruptions gracefully

### Demo Script (`demo.py`)
- Interactive text generation interface
- Shows token-by-token breakdown of generation
- Displays token probabilities and alternatives
- Visualizes token boundaries in generated text

## 📊 Features

### Tokenizer Training
- Byte-Pair Encoding (BPE) tokenization
- Configurable vocabulary size
- Progress tracking and state saving
- Handles dataset sampling for quick experiments

### Text Generation
- Temperature-controlled sampling
- Multiple samples per prompt
- Token probability visualization
- Top-k token alternatives display

### Visualization
- Token boundary markers
- Probability scores for each token
- Log probabilities for model decisions
- Raw token visualization with byte-level details

## 🚀 Getting Started

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Train the tokenizer:
   ```bash
   poetry run python main.py
   ```

3. Run the demo:
   ```bash
   poetry run python demo.py
   ```

## 🎮 Demo Commands

- `/help` - Show available commands
- `/params` - Adjust generation parameters
- `/tokens` - View vocabulary statistics
- `/quit` - Exit the demo

## 📝 Notes

- The model is trained on a small subset of TinyStories for quick experimentation
- Token boundaries are marked with `[]` in the demo output
- Each token shows its probability and top alternatives
- The tokenizer state is saved for resuming interrupted training

## 🔍 Example Output

```
Tokenization breakdown:
------------------------------------------------------------
Token ID   Token                Visualization
------------------------------------------------------------
71         he                   68 65
106        ll                   6c 6c
49         o                    6f
------------------------------------------------------------

Generated text with token boundaries:
[he][ll][o][ soft][ec][Why][ sil][ flower]

Clean text with natural spacing:
hello soft ec Why sil flower
```

## 🤝 Contributing

Feel free to experiment with:
- Different vocabulary sizes
- Alternative datasets
- New visualization methods
- Generation parameters

## 📚 Resources

- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [🤗 Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index) 