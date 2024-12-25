# LLM Graph Node Types

## Data Source Node
- Fetches data from various sources.
- Configuration:
  - Source type dropdown (file, API, etc.)
  - Source-specific settings (path, URL, etc.)
  - Chunk size (bytes/tokens)
- Events emitted: `{type: "dataSource", data: string | Buffer, metadata: { path?: string, totalChunks?: number, currentChunk?: number }}`
- Example usage:
  - File System: Reads files and directories.
  - API: Fetches data from a web service.

## Command Node
- Executes shell commands.
- Configuration:
  - Command input
- Events emitted: `{type: "command", output: string, error?: string}`
- Example usage:
  - `ls -lah`: Lists files in a directory.
  - `git status`: Checks the status of a git repository.

## LLM Processing Node
- Processes text using language models.
- Configuration:
  - Model selection dropdown
  - Prompt textarea
- Events emitted: `{type: "llm", chunk: string, complete: boolean}`
- Example usage:
  - Summarize text
  - Filter text
  - Generate text

## Filter Node
- Buffers and filters events based on JavaScript conditions.
- Configuration:
  - JavaScript condition code
  - Available variables:
    - `event`: Current event object
    - `buffer`: Array of buffered events
    - `time`: Milliseconds since last emit
- Events emitted: Same as input event, if condition is met.
- Example conditions:
  - `buffer.length >= 3` // Every 3 events
  - `time > 1000` // Every second
  - `event.chunk?.includes("hello")` // Contains text

## Memory Node
- Stores and retrieves conversation history.
- Configuration:
  - Memory type dropdown:
    - Short-term (clears on reload)
    - Long-term (persists in localStorage)
  - Memory format: JSON object
  - Max memory size (in entries)
  - Retrieval strategy:
    - FIFO (default)
    - Relevance (requires embedding)
- Events emitted:
  ```javascript
  {
    type: "memory",
    action: "store" | "retrieve",
    content: string,
    timestamp: number
  }
  ```

## Output Node
- Displays received events.
- Configuration:
  - Template string for formatting output
  - Default: `${event}` shows full JSON
  - Can access specific fields: `${event.type}`, `${event.chunk}`
- Example templates:
  - `${event}` // Full event JSON
  - `${event.chunk}` // Just the chunk content
  - `Received at ${event.time}: ${event.type}` // Custom format

## Leaky Bucket Node
- Regulates the flow of events, chunking large outputs and spawning temporary LLM nodes.
- Configuration:
  - Chunk size (bytes/tokens)
  - Rate limit (events per second)
  - LLM model for processing chunks
  - Prompt for processing chunks
- Events emitted: `{type: "leakyBucket", chunk: string, complete: boolean}`

## Merger Node
- Merges multiple input events into a single output using an LLM.
- Configuration:
  - LLM model for merging
  - Prompt for merging
- Events emitted: `{type: "merger", output: string, complete: boolean}`

## Common Features
- All nodes have:
  - Run/Stop button
  - Debug toggle (🐛)
  - Connection points
  - Draggable positioning
  - Saveable state