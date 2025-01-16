# States

A state represents the internal memory of a language model at a specific point in time. It encapsulates:

- **Model Configuration:** Parameters like temperature, top_p, and seed.
- **Context History:** The sequence of tokens processed so far.
- **Attention Mask:** Information about which tokens the model should attend to.
- **Cache:** Key-value pairs used for efficient generation.
- **Metadata:** Information about the state's creation and modifications.

States are central to the Cacher API, allowing for:

- **Context Management:** Continuing conversations and maintaining context over multiple turns.
- **Reproducibility:** Generating the same output given the same state and parameters.
- **Forking and Merging:** Creating variations of a state and combining them.
- **Analysis:** Inspecting the internal workings of the model at a specific state.
