# Friendly Input Field

## Overview

A browser-based input field that captures user events (mouse and keyboard) and processes them through local LLMs using Ollama. Built with WebSocket communication for real-time event processing and streaming responses.

## Current State

*   **Event Batch Processing**: Fixed a critical issue where event batches were not being processed due to double JSON stringification on the client-side. The system now correctly captures, batches, and sends events to the server for processing.
*   **Model Selection**: Implemented automatic model selection with a configurable prefix (currently set to 'smollm2'). The client filters available models based on this prefix and allows the user to select a model.
*   **WebSocket Communication**: Enhanced WebSocket communication for model selection and event processing. The server can now handle `getModels` and `event_batch` requests correctly.
*   **Streaming Responses**: The server streams responses from Ollama back to the client, providing a more interactive experience.
*   **Error Handling**: Improved error handling on both the client and server sides, including better JSON parsing and error reporting.
*   **Debug Logging**: Added extensive debug logging to track model selection, request handling, and WebSocket communication.

## Features

1. **Event Capture**
    *   Real-time keyboard and mouse event tracking.
    *   Event batching with a configurable buffer size.
    *   Automatic event flushing when the buffer is full or after a timeout.

2. **LLM Integration**
    *   Automatic model selection based on a prefix.
    *   Streaming responses from Ollama.
    *   Watchdog timer for stalled requests (currently set to 5 seconds).
    *   Multiple model support.

3. **Debug Features**
    *   Real-time model selection status.
    *   Event batch logging.
    *   WebSocket connection status.
    *   Detailed error reporting in the UI.

## Implementation Details

### Server Components

The server implementation is based on:

```javascript:friendly_input_field/server/server.js
startLine: 47
endLine: 127
```

Key features:

*   **Model Availability Checking**: The server can check for available models from Ollama.
*   **Request Handling with Timeouts**: Implements a watchdog timer to monitor and abort stalled requests.
*   **Streaming Response Processing**: Processes and streams responses from Ollama to the client.
*   **Error Handling with Client Feedback**: Sends error messages back to the client for display.

### Client Components

The client implementation includes:

```javascript:friendly_input_field/client/js/main.js
startLine: 1
endLine: 51
```

Features:

*   **Automatic Model Selection**: Filters and selects models based on a configurable prefix.
*   **Event Batching and Processing**: Captures, batches, and sends user input events to the server.
*   **Real-time UI Updates**: Updates the UI with model selection status, event logs, and streaming responses.
*   **Error Handling and Display**: Displays error messages received from the server.

## Setup

1. **Prerequisites**
    *   Node.js
    *   Ollama with required models installed
    *   WebSocket-capable browser

2. **Installation**

    ```bash
    npm install
    ```

3. **Configuration**
    *   Set `MODEL_PREFIX` in `main.js` to match your preferred model prefix.
    *   Adjust WebSocket connection parameters in `websocket.js` if needed.
    *   Configure event buffer size and timeout values in `eventCapture.js`.

4. **Running**

    ```bash
    npm start
    ```

## Current Limitations

*   Model selection requires exact prefix matching.
*   No retry mechanism for failed model selections.
*   Limited error recovery for stalled requests.

## Next Steps

1. Implement a model fallback chain.
2. Add a retry mechanism for failed requests.
3. Improve error handling and recovery.
4. Add a configuration UI for model selection.
5. Implement request queuing and rate limiting.

## Technical Notes

*   Uses WebSocket for bi-directional communication.
*   Implements a watchdog timer for request monitoring.
*   Handles streaming responses with proper cleanup.
*   Maintains debug logging for troubleshooting.
```

