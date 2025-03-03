# Investigating Token Probability with Hugging Face Models

## Objective
Develop a script to visualize the probability of the next token pairs generated by Hugging Face language models. This involves analyzing the model's predictions and comparing them to the actual next tokens in a given sentence.

## Key Functionality
- **Token Pair Probability Visualization:** The script calculates and displays the probabilities of the most likely next token pairs predicted by a given language model.
- **Actual Next Word Probability:** It also highlights the probability of the actual next word pair in the sentence, even if it's not among the top predicted pairs.
- **Raw Token Display:** The script shows the raw token IDs corresponding to both the predicted and actual word pairs.
- **Handling Tokenization Variations:** The script addresses the challenge of different tokenizations for the same word or phrase by considering multiple possible token ID sequences.
- **Context-Dependent Analysis:** The visualization is performed step-by-step through the input sentence, showing the model's predictions at each stage.

## Implementation Details
- **Hugging Face Transformers:** The implementation utilizes the Hugging Face Transformers library to load models and tokenizers, and to generate token probabilities.
- **Rich Library:** The `rich` library is used to create formatted tables for clear and organized output of the probability data.
- **Probability Calculation:** The script calculates the joint probability of consecutive token pairs by multiplying the probabilities of the individual tokens.
- **Tokenization Analysis:** A helper function is included to analyze and display different possible tokenizations for a given text.

## Testing and Verification
A Python script (`test_generate_chat.py`) was developed to test and refine the token probability visualization. The script includes the following key functionalities:

1. **Loading Model and Tokenizer:** The script loads a specified Hugging Face model and its corresponding tokenizer.
2. **Visualizing Probabilities:** The `visualize_actual_next_word_probabilities` function iterates through a given sentence, calculating and displaying the probabilities of the next token pairs at each step.
3. **Top N Predictions:** The `get_top_n_token_pairs` function retrieves the top N most probable next token pairs predicted by the model.
4. **Actual Next Word Probability:** This function also calculates the probability of the actual next word pair in the sentence.
5. **Token ID Display:** The raw token IDs for both predicted and actual word pairs are displayed in the output tables.
6. **Handling Tokenization Variations:** The `normalize_tokens` function identifies and handles different possible tokenizations for the same text, ensuring accurate probability calculations.
7. **Tokenization Analysis:** The `print_token_info` function provides detailed information about how a given text is tokenized, including different variations based on casing and spacing.

### Challenges and Solutions
During the development and testing phase, several challenges were encountered and addressed:

- **Missing Probability for Actual Next Words:** Initially, the script did not always display the probability of the actual next word pair if it was not among the top predicted pairs. This was resolved by explicitly calculating and displaying this probability, even if it was very low.
- **Incorrect Matching of Actual Next Words:** The initial implementation compared raw token IDs, which led to mismatches when the same word was tokenized differently. This was fixed by comparing the decoded text of the token pairs instead of the raw token IDs.
- **Handling Different Tokenizations:** The model sometimes uses different token IDs for the same word in different contexts. To address this, the `normalize_tokens` function was implemented to identify all possible tokenizations of the actual next words, and the script was modified to consider all these variations when calculating probabilities.
- **Inconsistent Token IDs for the Same Word:** It was observed that the model's tokenizer could produce different token IDs for the same word (e.g., "want") depending on the context (e.g., with or without a leading space). The `print_token_info` function was added to analyze these variations and the `get_top_n_token_pairs` function was enhanced to handle these differences.
- **Low Probability for Actual Next Words:** In some cases, the probability of the actual next words was very low or even zero according to the initial calculations. This was addressed by ensuring that all possible tokenizations of the actual next words were considered when calculating their probability.
- **Displaying Raw Tokens:** The user requested the display of raw token IDs in the output tables. This was implemented by adding a "Token IDs" column to the tables.
- **Probability Not Showing in Tables:** After implementing the handling of tokenization variations, the probability of the actual next words was sometimes missing from the output tables. This was resolved by ensuring the probability was always calculated and displayed, even if it was zero.

Through iterative testing and debugging, these issues were resolved to ensure the accurate and informative visualization of token probabilities.

## Status
The script for visualizing token pair probabilities has been successfully implemented and refined. It now accurately displays the probabilities of predicted token pairs and the actual next word pairs, handling various tokenization scenarios. The `test_generate_chat.py` script provides a robust way to test and understand the model's token prediction behavior.

## Benefits
- **Understanding Model Predictions:** Provides insights into how language models predict the next tokens in a sequence.
- **Debugging Tokenization Issues:** Helps identify and understand different tokenization strategies used by the model.
- **Educational Tool:** Useful for learning about the inner workings of language models and their probabilistic nature.

For more information on the underlying mechanisms, refer to the [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/main/en/model_doc/llama3).




## sample output

Token analysis for 'want to':
Token IDs: [43667, 288]
Tokens: ['want', 'Ġto']
Different tokenization for 'WANT TO': [71, 17321, 9549]
Different tokenization for ' want to': [1277, 288]
Different tokenization for 'want to ': [43667, 288, 216]

After: 'AI assistant. User: I'
Actual next words: 'want to' (tokens: [43667, 288])
           Next Token Pair Probabilities           
┏━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Rank ┃ Token Pair   ┃ Token IDs   ┃ Probability ┃
┡━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│   #1 │ 'need to'    │ [737, 288]  │      0.1047 │
│   #2 │ 'have a'     │ [457, 253]  │      0.0818 │
│   #3 │ 'want to'    │ [1277, 288] │      0.0526 │
│   #4 │ 'would like' │ [736, 702]  │      0.0259 │
│   #5 │ 'need help'  │ [737, 724]  │      0.0250 │
└──────┴──────────────┴─────────────┴─────────────┘


Token analysis for 'to understand':
Token IDs: [1141, 1044]
Tokens: ['to', 'Ġunderstand']
Different tokenization for 'TO UNDERSTAND': [10179, 4954, 29738, 3127, 12386]
Different tokenization for ' to understand': [288, 1044]
Different tokenization for 'to understand ': [1141, 1044, 216]

After: 'AI assistant. User: I want'
Actual next words: 'to understand' (tokens: [1141, 1044])
             Next Token Pair Probabilities             
┏━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Rank ┃ Token Pair      ┃ Token IDs    ┃ Probability ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│   #1 │ 'to know'       │ [288, 699]   │      0.1263 │
│   #2 │ 'to write'      │ [288, 2965]  │      0.1162 │
│   #3 │ 'to create'     │ [288, 1464]  │      0.0635 │
│   #4 │ 'to start'      │ [288, 1120]  │      0.0568 │
│   #5 │ 'you to'        │ [346, 288]   │      0.0544 │
│  --- │ ---             │ ---          │         --- │
│  N/A │ 'to understand' │ [1141, 1044] │      0.0000 │
└──────┴─────────────────┴──────────────┴─────────────┘





# Token Prediction Analysis Tool

A tool for analyzing and visualizing how language models predict sequences of tokens, with special focus on early stopping behavior and probability distributions.

## Key Findings

### 1. End Token Behavior
- Models have natural stopping points using special tokens (EOS, SEP) and punctuation
- The model recognized `<|im_end|>` as a strong completion token (probability 0.131625)
- Short, complete sequences often have higher probabilities than longer ones

### 2. Sequence Patterns
Observed distinct types of completions:
- **Short Commands**: Single tokens like 'User' (prob: 0.121381)
- **Natural Endings**: "User: Thanks.<|im_end|>" (prob: 0.003753)
- **Continuation Attempts**: Longer sequences that try to start new dialogue turns

### 3. Probability Distribution
- Short, definitive responses have highest probabilities (0.13-0.12 range)
- Medium-length completions fall in middle range (0.003-0.001)
- Long, exploratory sequences have lowest probabilities (<0.001)

### 4. Beam Search Statistics
From sample run:
```
Total Sequences Explored: 425
Completed Sequences: 56
Early Stops (Special Tokens): 5
Early Stops (Punctuation): 0
Pruned Sequences: 326
```

## Technical Implementation

### End Token Handling
```python
# Special tokens and punctuation are tracked separately
end_token_ids = set()
special_token_ids = set()

# Add model's special tokens
if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
    end_token_ids.add(tokenizer.eos_token_id)
    special_token_ids.add(tokenizer.eos_token_id)
```

### Probability Calculation
- Uses beam search with configurable width
- Tracks both special token and punctuation stops
- Maintains running probabilities for sequences

### Visualization
- Rich tables showing:
  - Sequence content
  - Token IDs
  - Completion status
  - Probability scores
  - Sequence length

## Usage

```python
visualize_next_token_sequences(
    tokenizer,
    model,
    sentence="System: You are a helpful assistant. Assistant: Hello. User: ",
    max_tokens=10,
    top_n_best=10,
    top_k_beam_width=5,
    temperature=1.0
)
```

## Future Improvements

1. **Enhanced Early Stopping**
   - Add more natural language end points
   - Weight different types of stops differently

2. **Probability Analysis**
   - Add entropy calculations
   - Track probability distribution patterns
   - Analyze token correlation patterns

3. **Visualization**
   - Add probability distribution graphs
   - Visualize beam search tree
   - Show token relationships

## Dependencies
- transformers
- torch
- rich (for visualization)