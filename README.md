# [epub_gpt.js](epub_gpt.js)

## Project Overview

This project involves processing an EPUB book's structure using an AI model to generate summaries for each chapter. The AI maintains context between chapters, ensuring that summaries build on previous content. The process involves handling various content types, managing the AI's context window, and providing detailed logging of the processing steps.

## Features

### 1. **EPUB Structure Processing**
- Parses the EPUB book structure into JSON format.
- Processes each chapter individually, handling text and non-text content differently.

### 2. **Non-Text Content Handling**
- Skips non-text content like images and SVGs, logging a trimmed version to understand what was skipped.
- Ensures that the processing continues smoothly even after encountering non-text content.

### 3. **Context Management**
- Accumulates context between chapters to enable the AI to generate summaries that are aware of previous content.
- Manages the AI’s context window (4096 tokens) to avoid exceeding the model’s token limit.

### 4. **Detailed Logging**
- Logs important details such as input length, processing time, token generation speed, and cost.
- Uses colored logging for better readability and information differentiation.

## How It Works

1. **Load Book Structure**: The JSON representation of the book structure is loaded from a file.
2. **Process Chapters**: Each chapter is processed, and its content is sent to the AI model for summarization.
    - **Context Accumulation**: Summaries from previous chapters are accumulated and included in the prompt for the next chapter.
    - **Trimming**: The combined prompt is trimmed to fit within the model's context window.
3. **Logging**: The script logs the AI's response, processing speed, and other metrics in a structured, color-coded format.

## Setup and Requirements

- **Node.js**: Ensure Node.js is installed.
- **gpt4all**: The project uses the gpt4all package for AI model interaction.
- **chalk**: Used for colored logging in the console.

### Installation

1. Clone the repository.
2. Install the required packages using `npm install` or `yarn install`.
3. Place your EPUB file in the appropriate directory and adjust the path in the script.

### Running the Script

```bash
node epub_gpt.js
```

This will start the processing, and the script will log the progress and summaries in the console.

## Future Enhancements

- **Token Management**: Refine context management to ensure optimal use of the token window.
- **Performance**: Explore GPU utilization for faster processing.
- **Expanded Context**: Develop strategies for handling larger contexts or summarizing effectively.

---