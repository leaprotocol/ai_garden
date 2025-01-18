---
title: "Adventures in Tokenizer Training: Building a Tiny Language Model from Scratch"
date: 2024-01-18
public: true
tags: [machine-learning, nlp, python, tokenizer, language-model]
---

# Adventures in Tokenizer Training: Building a Tiny Language Model from Scratch

## The Journey Begins 🚀

What started as a simple experiment in training a tokenizer turned into a fascinating deep dive into the internals of language models. Today, I want to share our journey of building and debugging a small language model using the TinyStories dataset.

## Why TinyStories? 📚

The [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) is perfect for experimentation - it's small enough to train quickly but large enough to learn meaningful patterns. Plus, its simple vocabulary and straightforward narratives make it easier to spot when things go wrong.

## The First Challenge: Tokenizer Training 🔍

Our first hurdle was getting the tokenizer training right. We started with a vocabulary size of 8192 tokens, using Byte-Pair Encoding (BPE). The initial implementation had a few interesting bugs:

```python
if processed_count >= total_size:
    # This was our first bug - it didn't account for the last batch!
    return tokenizer
```

We fixed this by adjusting the completion detection:

```python
if processed_count >= total_size - batch_size:
    # Now we properly handle the last batch
    return tokenizer
```

## Debugging Through Visualization 🎨

One of the most enlightening parts was adding detailed token visualization. Here's what we discovered:

```
Token breakdown:
[he][ll][o][ soft][ec][Why][ sil][ flower]
```

This revealed that our tokenizer was:
1. Correctly identifying common prefixes ("he", "ll", "o")
2. Sometimes making strange splits ("soft" + "ec")
3. Preserving word boundaries with the Ġ character

## The Probability Dance 🎲

We added probability visualization to understand the model's choices:

```
Token 3:
  Text: 'soft'
  Probability: 0.0234
  Log Probability: -3.7551

Top alternatives:
1. 'the' - Prob: 0.0456
2. 'and' - Prob: 0.0234
3. 'was' - Prob: 0.0189
```

This helped us understand why the model sometimes made unexpected choices - it was often choosing between several equally plausible tokens.

## Making It Interactive 🎮

The final piece was building an interactive demo that shows the generation process in real-time. This turned out to be invaluable for debugging and understanding the model's behavior.

```python
def show_token_details(model, tokenizer, input_ids, token_id, position):
    """Show detailed information about a generated token."""
    with torch.no_grad():
        outputs = model(input_ids[:, :position+1])
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # ... visualization code ...
```

## Lessons Learned 📝

1. **Visibility is Key**: Adding detailed logging and visualization made debugging much easier
2. **Token Boundaries Matter**: Understanding how the tokenizer splits text is crucial
3. **State Management**: Proper handling of training state allows for interrupted and resumed training
4. **Temperature Control**: The generation temperature dramatically affects output coherence

## The Results 🎯

After all our debugging and improvements, we ended up with a system that can:
- Train a custom tokenizer from scratch
- Generate text with visible token boundaries
- Show probability distributions for each token
- Handle training interruptions gracefully

Here's a sample of what it can do:

```
Input: "hello"

Raw tokens:
he ll o Ġsoft Ġwhy Ġflower

Clean output:
hello soft why flower
```

## What's Next? 🌟

There's still so much to explore:
- Training on larger datasets
- Experimenting with different vocabulary sizes
- Adding more visualization tools
- Implementing attention visualization

## Try It Yourself! 🔧

The code is available in our repository. To get started:

```bash
poetry install
poetry run python demo.py
```

Feel free to experiment and share your findings!

## Technical Details 🔬

For those interested in the nitty-gritty:
- Tokenizer: Byte-Pair Encoding with vocab size 8192
- Model: Small GPT-2 style architecture
- Training: Using 🤗 Transformers and Tokenizers libraries
- Visualization: Custom token boundary and probability displays

## Conclusion 🌈

Building and debugging a tokenizer from scratch was a challenging but rewarding experience. It gave us deep insights into how language models work at the token level and helped us build better tools for understanding their behavior.

Remember: the best way to learn is to experiment, break things, and fix them! Happy coding! 🚀 