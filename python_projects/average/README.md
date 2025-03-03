# Language Model Analysis and Experimentation

This project contains tools for analyzing and experimenting with language models, particularly the SmolLM2 series. It includes functionality for:

1. **Prompt Averaging**: Weighted combination of multiple prompts' embeddings for text generation
2. **SWAG Dataset Evaluation**: Testing model performance on multiple-choice sentence completion
3. **LAMBADA Dataset Evaluation**: Assessing model's ability to predict final words in sentences
4. **Interactive Generation**: Command-line interface for experimenting with text generation

## Key Features

### Prompt Averaging (`main.py`)
- Weighted combination of multiple prompts' embeddings
- Interactive command-line interface
- Flexible weighting system
- Detailed tokenization visualization

### SWAG Evaluation (`test_swag.py`)
- Dual model setup (evaluation and training)
- Position-wise token probability analysis
- Multiple scoring methods (normalized, average, median, worst)
- Detailed logging and metrics tracking

### LAMBADA Evaluation (`test_lambada.py`)
- Target word probability analysis
- BERT-based context validity scoring
- Top-k prediction tracking
- Comprehensive metrics (success rate, target probability, rank)

## Usage

### Interactive Generation
```bash
python main.py
```

### SWAG Evaluation
```bash
python test_swag.py
```

### LAMBADA Evaluation
```bash
python test_lambada.py
```

## Results and Analysis

See [blogpost.md](blogpost.md) for detailed analysis of our findings and experimental results.