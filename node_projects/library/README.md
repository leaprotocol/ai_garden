Okay, here's a draft of the `README.md` documentation for the `@library` package, incorporating the recent developments and focusing on the Ollama client, streaming, and examples:

# @library

This package provides a JavaScript client library for interacting with the Ollama API, along with examples demonstrating its usage for streaming language model responses, handling token events, and managing multiple conversations.

## Ollama Client (`ollama_client.js`)

The `ollama_client.js` module provides the core functionality for interacting with the Ollama API. It exports the following:

### `getAvailableModels()`

Fetches a list of available models from the Ollama server.

**Returns:**

-   `Promise<Array<object>>`: An array of model objects, each containing:
    -   `name` (string): The name of the model.
    -   `size` (number): The size of the model in bytes.
    -   `modified` (string): The last modified timestamp.
    -   `digest` (string): The model's digest.

### `OllamaStreamHandler`

A class that handles streaming responses from the Ollama API. It extends the `EventEmitter` class and emits events based on the received data.

**Constructor:**

-   `constructor(stream, client)`:
    -   `stream`: The asynchronous iterable stream returned by `ollamaClient.generate()`.
    -   `client`: The `Ollama` client instance used to generate the stream.

**Properties:**

-   `stream`: The stream object.
-   `client`: The Ollama client instance.
-   `tokenCount` (number): The total number of tokens received in the current stream.
-   `currentMessage` (string): The accumulated text response from the LLM.

**Methods:**

-   `setupListeners()`: Sets up listeners for the stream's data events.
-   `onTokenCount(count, listener)`: Registers a listener function to be called when the token count reaches a specific value. The listener receives the current message as an argument.
-   `getCurrentMessage()`: Returns the current accumulated message.
-   `abort()`: Aborts the ongoing stream by calling `this.client.abort()`.

**Events:**

-   `data`: Emitted when a new chunk of data is received. The event payload is the raw data chunk.
-   `token`: Emitted when a new token is received. The event payload is the token string.
-   `tokenCount`: Emitted when the token count is updated. The event payload is the new token count.
-   `end`: Emitted when the stream is complete.
-   `error`: Emitted when an error occurs. The event payload is the error object.
-   `abort`: Emitted when the stream is aborted.

### `generateStream(model, prompt, options = {})`

Generates a streaming response from the Ollama API.

**Parameters:**

-   `model` (string): The name of the model to use.
-   `prompt` (string): The prompt to send to the model.
-   `options` (object, optional): Additional options to pass to the `ollamaClient.generate()` method.

**Returns:**

-   `Promise<OllamaStreamHandler>`: An instance of `OllamaStreamHandler` that manages the stream.

### `generate(model, prompt, options = {})`

Generates a non-streaming response from the Ollama API.

**Parameters:**

-   `model` (string): The name of the model to use.
-   `prompt` (string): The prompt to send to the model.
-   `options` (object, optional): Additional options to pass to the `ollamaClient.generate()` method.

**Returns:**

-   `Promise<object>`: The response object from the Ollama API.

## Examples

The `examples` directory contains several examples demonstrating how to use the `@library` package.

### `stream_text.js`

Demonstrates basic text streaming from an Ollama model.

```javascript:library/examples/stream_text.js
startLine: 1
endLine: 26
```

-   Uses `generateStream` to create a stream.
-   Listens for `token` events to print tokens to the console.
-   Listens for `end` and `error` events to handle stream completion and errors.

### `token_event.js`

Demonstrates handling token events and aborting the stream.

```javascript:library/examples/token_event.js
startLine: 1
endLine: 48
```

-   Uses `generateStream` to create a stream.
-   Listens for `token` events to print tokens to the console.
-   Uses `onTokenCount` to log the full message when the token count reaches 20.
-   Uses `setInterval` to periodically log the current message.
-   Aborts the stream when the token count reaches 40 using `streamHandler.abort()`.
-   Listens for `abort`, `end`, and `error` events to handle stream termination and errors.

### `token_event2.js`

Demonstrates a multi-user philosopher debate using WebSockets.

```javascript:library/examples/token_event2.js
startLine: 1
endLine: 148
```

-   Sets up a WebSocket server using `ws`.
-   Serves `index.html` and `styles.css`.
-   Implements a `conversationBuilder` function to manage the dialogue history.
-   Uses `generateStream` to create streams for each philosopher (Hegel and Marcus Aurelius).
-   Listens for `data`, `abort`, `end`, and `error` events to manage the conversation flow.
-   Sends messages to the client via WebSockets to update the UI.
-   Aborts the stream after 70 tokens.
-   Continues the dialogue for 6 turns.

### `multi_conversation.js`

Demonstrates handling multiple concurrent conversations using WebSockets.

```javascript:library/examples/multi_conversation.js
startLine: 1
endLine: 97
```

-   Sets up a WebSocket server.
-   Serves `index_multi.html`.
-   Starts multiple conversations (3 in this example) using `generateStream`.
-   Sends messages to the client via WebSockets to update the UI for each conversation.
-   Handles `data`, `end`, and `error` events for each stream.

### `linux_tools.js`

Demonstrates using an LLM as a Linux command-line tool using WebSockets.

```javascript:library/examples/linux_tools.js
startLine: 1
endLine: 184
```

-   Sets up a WebSocket server.
-   Serves `index_multi.html`.
-   Whitelists allowed Linux commands (`ls`, `pwd`, `date`, `whoami`, `uptime`).
-   Uses `generateStream` with a system prompt that instructs the LLM to act as a Linux command-line tool.
-   Parses the LLM's response to extract tool calls (e.g., `{"tool": "linux_command", "command": "ls"}`).
-   Executes the command using `child_process.exec` if it's allowed.
-   Sends the command result or error back to the client via WebSockets.
-   Handles `data`, `end`, and `error` events.

### `index.html`

The HTML file for `token_event2.js`, providing a simple UI for the philosopher debate.

```html:library/examples/index.html
startLine: 1
endLine: 85
```

### `index_multi.html`

The HTML file for `multi_conversation.js` and `linux_tools.js`, providing a UI for multiple conversations.

```html:library/examples/index_multi.html
startLine: 1
endLine: 126
```

### `styles.css`

A CSS file providing basic styling for `index.html`.

```css:library/examples/styles.css
startLine: 1
endLine: 26
```

## Running the Examples

1. Make sure you have Ollama installed and running.
2. Install the dependencies:

    ```bash
    npm install
    ```
3. Run the desired example:

    ```bash
    node examples/<example_name>.js
    ```

    For example:

    ```bash
    node examples/token_event.js
    ```

    For the WebSocket examples (`token_event2.js`, `multi_conversation.js`, `linux_tools.js`), open the corresponding HTML file in your browser after starting the server.

## Future Enhancements

-   **Error Handling:** Improve error handling and reporting.
-   **Logging:** Add more detailed logging for debugging and monitoring.
-   **Configuration:** Allow more configuration options for the Ollama client and stream handler.
-   **Testing:** Add unit tests for the `ollama_client.js` module.
-   **Documentation:** Add more detailed documentation for the API and examples.
-   **Security:** Implement security measures for the WebSocket server, such as authentication and authorization.
-   **Scalability:** Consider how to scale the solution for a large number of concurrent users and conversations.
-   **Extensibility:** Design the system to be easily extensible with new features and functionalities.
