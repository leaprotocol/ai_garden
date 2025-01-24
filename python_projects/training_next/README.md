# Training Next

An improved version of the training pipeline with better async support, interrupt handling, and testing.

## Features

- Asynchronous training pipeline with proper interrupt handling
- Modular architecture with separate components for dataset, tokenizer, and model training
- Comprehensive test suite
- Example demo scripts for various experiments
- Improved logging and error handling
- Bidirectional sequence analysis tools

## Project Structure

```
training-next/
├── dataset_utils.py     # Dataset loading and preprocessing
├── tokenizer_trainer.py # Tokenizer training functionality
├── model_trainer.py     # Model training functionality
├── main.py             # Main training script
├── demo.py             # Basic usage example
├── demo2.py           # LAMBADA dataset evaluation
├── demo3.py           # BERT masked token prediction
├── demo4.py           # Token presence analysis
├── demo5.py           # Reversed LAMBADA evaluation
├── demo6.py           # Bidirectional sequence analysis
└── tests/             # Test suite
    ├── conftest.py
    ├── test_dataset_utils.py
    ├── test_tokenizer_trainer.py
    └── test_model_trainer.py
```

## Experiments and Analysis

Our project evolved through several experiments investigating language model behavior:

1. **Basic Generation (demo.py)**
   - Initial text generation with probability tracking
   - Analysis of token-level probabilities and entropy

2. **LAMBADA Dataset Evaluation (demo2.py)**
   - Evaluated model performance on long-range dependencies
   - Compared predictions with SmolLM models
   - Analyzed prediction accuracy and confidence

3. **BERT Masked Prediction (demo3.py)**
   - Implemented masked token prediction using BERT
   - Analyzed prediction confidence and alternatives
   - Tested with various domain-specific examples

4. **Token Presence Analysis (demo4.py)**
   - Analyzed presence of target tokens in input context
   - Evaluated single vs multi-token targets
   - Measured context length impact

5. **Reversed LAMBADA (demo5.py)**
   - Explored bidirectional context influence
   - Analyzed prediction quality with limited context

6. **Bidirectional Analysis (demo6.py)**
   - Comprehensive bidirectional probability analysis
   - Visualization of forward/backward probabilities
   - Statistical analysis of contextual influence
   - Identification of surprising words and patterns

### Key Findings

- Models show different prediction patterns in forward vs backward contexts
- Context length significantly impacts prediction confidence
- Some words show surprisingly high probabilities in specific contexts
- Bidirectional analysis reveals asymmetric contextual dependencies
- Neighbor-only vs full-context comparisons highlight importance of long-range dependencies

## Installation

1. Make sure you have Python 3.8+ installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Demos

Basic generation:
```bash
python demo.py
```

Bidirectional analysis:
```bash
python demo6.py
```

Each demo script demonstrates different aspects of language model behavior and analysis.

### Running Tests

```bash
pytest tests/
```