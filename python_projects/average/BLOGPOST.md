# Exploring Language Model Capabilities: From Prompt Averaging to Dataset Evaluation

## Introduction

In this series of experiments, we explored various aspects of language model behavior using the SmolLM2 models. Our investigation spanned from creative text generation techniques to rigorous dataset evaluation.

## Key Experiments

### 1. Prompt Averaging with Weighted Embeddings

We implemented a novel text generation approach that combines multiple prompts through weighted averaging of their embeddings. This allows for:

- Blending of different concepts or styles
- Fine-grained control over prompt influence
- Creative exploration of prompt interactions

**Technical Implementation:**
- Weighted sum of prompt embeddings
- Automatic padding and normalization
- Interactive command-line interface

**Example Use Case:**
```python
generate_text_from_weighted_embeddings(
    prompts=['The cat sat on the', 'The dog ran in the', 'The bird flew over the'],
    weights=[0.4, 0.4, 0.2],
    max_length=60
)
```

### 2. SWAG Dataset Evaluation

We developed a comprehensive evaluation framework for the SWAG dataset, featuring:

- Dual model setup (evaluation and training)
- Position-wise token probability analysis
- Multiple scoring methods
- Detailed logging and metrics tracking

**Key Metrics:**
- Normalized probability accuracy
- Average log probability
- Median log probability
- Worst log probability

**Findings:**
- Models show varying performance across different scoring methods
- Position-wise analysis reveals interesting patterns in token prediction
- Training on specific examples can improve model performance

### 3. LAMBADA Dataset Analysis

Our LAMBADA evaluation focused on the model's ability to predict final words in sentences, including:

- Target word probability tracking
- BERT-based context validity scoring
- Top-k prediction analysis
- Comprehensive metrics collection

**Key Metrics:**
- Success rate (target in top 10 predictions)
- Average target probability
- Median rank of target word
- Context validity scores

**Findings:**
- Models perform better on common words than rare ones
- Context length significantly impacts prediction quality
- BERT-based scoring provides valuable insights into prediction validity

## Conclusion

These experiments demonstrate the versatility of language models and the importance of rigorous evaluation. The tools developed in this project provide valuable insights into model behavior and offer new ways to interact with and improve language models.