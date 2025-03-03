"""
Infinite Tokens
==============

A Python package for infinitely generating and streaming tokens from language models.

This package provides a simple interface for generating an infinite stream of tokens
from any Hugging Face compatible language model. It includes utilities for real-time
streaming, configuration, and example applications.

Main Components:
---------------
- InfiniteTokenGenerator: Core class that handles model loading and token generation
- Example applications showcasing different uses of infinite token generation

Example Usage:
-------------
```python
from infinite_tokens.main import InfiniteTokenGenerator

generator = InfiniteTokenGenerator()
generator.generate_infinite_tokens(initial_prompt="Once upon a time")
```
"""

from . import main
from .main import InfiniteTokenGenerator

__version__ = "0.1.0"
__all__ = ["InfiniteTokenGenerator", "main"] 