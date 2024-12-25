```markdown
# LLM Graph 7

LLM Graph 7 is an iteration of the LLM Graph project, featuring a React Flow-based UI with server-side node handling. This project enables visual creation and connection of various node types for building AI-powered workflows.

## Features

- **Base Node Class**: `AbstractNode` for common node functionalities
- **React Flow UI**: Interactive canvas for node placement and connections
- **Server-Side Nodes**: Nodes handle their events on the server and communicate statuses back to the client
- **WebSocket Communication**: Real-time event handling between server and client
- **Ollama Integration**: Direct connection to local Ollama instance for LLM capabilities
- **Debug Mode**: Per-node debug output for monitoring events and data flow

## Setup

### Prerequisites

- Node.js (v14 or later)
- npm or yarn
- Ollama running locally (default: http://localhost:11434)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/llm-graph7.git
   cd llm-graph7
   ```

2. **Install Dependencies**

   ```bash
   npm install
   # or
   yarn install
   ```

3. **Run the Server**

   ```bash
   npm run server
   # or
   yarn server
   ```llm-graph7/CONTEXT.md

4. **Run the Client**

   In a separate terminal:

   ```bash
   npm run dev
   # or
   yarn dev
   ```

   Open your browser and navigate to `http://localhost:5173`.

## Node Types

### LLM Node

- **Purpose**: Process text using local language models via Ollama
- **Features**:
  - Dynamic model selection from available Ollama models
  - Real-time streaming responses
  - Debug mode for monitoring token generation
  - Connection support for chaining with other nodes
- **Configuration**:
  - Model selection dropdown
  - Prompt input textarea
  - Process button with loading state
  - Debug toggle

### Timer Node

- **Purpose**: Emits timed events at configurable intervals
- **Features**:
  - Start/Stop controls
  - Configurable interval in milliseconds
  - Emits events with timestamp and formatted time
  - Debug mode for monitoring ticks

### Output Node

- **Purpose**: Displays and logs received events
- **Features**:
  - Activate/Deactivate toggle
  - Clear events button
  - Debug mode for monitoring incoming events
  - Connection validation
  - Event history with timestamps and sources
  - Formatted display of different event types

## Planned Node Types

### Memory Node (Planned)

- **Purpose**: Store conversation history or arbitrary data
- **Features**:
  - Configurable memory types (short-term, long-term)
  - Maximum size limits
  - Different retrieval strategies (FIFO, etc.)
  - Persistent storage option
- **Use Cases**:
  - Maintaining context between LLM calls
  - Storing intermediate results
  - Building conversational flows

### Merger Node (Planned)

- **Purpose**: Combine multiple inputs using LLM processing
- **Features**:
  - Multiple input handles
  - Aggregation logic configurable via prompts
  - Emits merged output events
- **Use Cases**:
  - Summarizing content from multiple sources
  - Merging data streams into a cohesive output

### Condition Node (Planned)

- **Purpose**: Flow control and event filtering based on conditions
- **Features**:
  - JavaScript condition evaluation
  - Branching based on condition outcomes (`true` or `false`)
  - Looping mode for iterative processing
- **Use Cases**:
  - Filtering events based on sentiment or content
  - Dynamic routing of events in workflows

### Data Source Node (Planned)

- **Purpose**: Integrate external data sources into workflows
- **Features**:
  - Support for various data sources (APIs, databases, webhooks)
  - Configurable fetch intervals and parameters
  - Emits data events for downstream processing
- **Use Cases**:
  - Fetching real-time data for analysis
  - Integrating third-party services into workflows

## Innovative Use Cases

LLM Graph 7's event-driven, node-based architecture offers a versatile foundation for various applications. Below is a comprehensive list of practical applications:

### Content & Media

1. Generate daily social media posts with automated A/B testing of engagement
2. Create automated podcast show notes from audio transcripts
3. Build a content repurposing pipeline (blog → social → newsletter)
4. Generate and schedule localized content for different time zones
5. Create automated video script outlines from trending topics
6. Build a real-time news aggregator and summarizer with sentiment analysis
7. Generate product descriptions from technical specifications
8. Create automated documentation updates from git commits
9. Build a multi-source research paper summarization system
10. Generate SEO-optimized meta descriptions for website content

### Development & Testing

11. Create automated code review assistant with best practice suggestions
12. Generate unit tests from code changes
13. Build a bug report analyzer and priority sorter
14. Create automated API documentation from endpoint usage
15. Generate code examples from documentation
16. Build a commit message quality checker and improver
17. Create automated release notes from git history
18. Generate code refactoring suggestions based on static analysis
19. Build a dependency update impact analyzer
20. Create automated PR description generator

### Business & Analytics

21. Build a customer feedback categorization and response system
22. Create automated competitive analysis reports from web scraping
23. Generate sales email templates based on prospect data
24. Build a real-time market sentiment analyzer
25. Create automated financial report summaries
26. Generate meeting minutes from audio transcripts
27. Build a customer support ticket prioritization system
28. Create automated expense report categorization
29. Generate performance review summaries from multiple data sources
30. Build a lead scoring system based on multiple criteria

### Education & Learning

31. Create personalized study guides from course materials
32. Build an automated quiz generator from educational content
33. Generate simplified explanations of complex topics
34. Create language learning exercises from real-world content
35. Build a homework helper that explains step-by-step solutions
36. Generate flashcards from lecture notes
37. Create educational content adaptations for different age groups
38. Build a concept relationship mapper for study topics
39. Generate practice problems with varying difficulty levels
40. Create automated progress reports for students

### Research & Analysis

41. Build a research paper recommendation system based on reading history
42. Create automated literature review summaries
43. Generate research methodology comparisons
44. Build a citation network analyzer
45. Create automated experiment log summarizers
46. Generate research proposal drafts from project notes
47. Build a data analysis report generator
48. Create automated patent analysis reports
49. Generate research gap identification reports
50. Build a cross-discipline research connector

### Personal Productivity

51. Create an email summarizer and priority sorter
52. Build a personal knowledge base curator
53. Generate weekly productivity reports from activity logs
54. Create automated task prioritization based on multiple inputs
55. Build a meeting scheduler with context-aware scheduling
56. Generate personal development plans from goal tracking
57. Create automated diary entries from daily activities
58. Build a habit tracking analyzer with improvement suggestions
59. Generate personalized reading recommendations
60. Create automated time-blocking schedules

### Data Processing

61. Build a data cleaning and normalization pipeline
62. Create automated data quality reports
63. Generate data transformation rules from examples
64. Build a multi-source data aggregator with custom merging
65. Create automated data validation checks
66. Generate data visualization suggestions
67. Build a data anomaly detection system
68. Create automated dataset documentation
69. Generate data sampling strategies
70. Build a data versioning system with change summaries

### AI/ML Operations

71. Create automated model performance reports
72. Build a training data quality analyzer
73. Generate model behavior comparison reports
74. Create automated model documentation
75. Build a model prediction explainer
76. Generate feature importance analysis
77. Create automated model retraining triggers
78. Build a model version comparison system
79. Generate model bias analysis reports
80. Create automated model deployment validation

### Integration & Automation

81. Build a multi-API workflow orchestrator
82. Create automated API integration tests
83. Generate webhook handling logic
84. Build a service health monitoring system
85. Create automated system integration reports
86. Generate API usage optimization suggestions
87. Build a service dependency mapper
88. Create automated failover procedures
89. Generate integration documentation
90. Build a cross-service error correlation system

### Monitoring & Debugging

91. Create automated error pattern analysis
92. Build a system performance bottleneck detector
93. Generate debug log summaries
94. Create automated incident reports
95. Build a real-time system health dashboard
96. Generate performance optimization suggestions
97. Create automated recovery procedures
98. Build a user impact analyzer for system issues
99. Generate system stability reports
100. Create automated alert correlation analysis

## Recommended New Node Types

To further enhance the functionality of LLM Graph 7, the following new node types are recommended:

### 1. Condition Node

- **Purpose**: Flow control and event filtering based on conditions
- **Features**:
  - JavaScript condition evaluation
  - Branching based on condition outcomes (`true` or `false`)
  - Looping mode for iterative processing
- **Use Cases**:
  - Filtering events based on sentiment or content
  - Dynamic routing of events in workflows

### 2. Database Node

- **Purpose**: Data persistence and database interactions
- **Features**:
  - Support for SQL and NoSQL databases
  - Parameterized queries to prevent injection attacks
  - Connection management with pooling and reconnections
- **Use Cases**:
  - Storing and retrieving workflow data
  - Integrating with external databases for data-driven workflows

### 3. APIRequest Node

- **Purpose**: External API integration
- **Features**:
  - Supports various HTTP methods (GET, POST, PUT, DELETE)
  - Handles authentication (API keys, OAuth tokens)
  - Automatically parses JSON, XML, or other response formats
- **Use Cases**:
  - Fetching data from third-party APIs
  - Sending data to external services as part of workflows

### 4. Transform Node

- **Purpose**: Data transformation and processing
- **Features**:
  - Custom transformation logic via JavaScript functions
  - Chainable transformations for complex processing
  - Error handling to catch and emit transformation failures
- **Use Cases**:
  - Manipulating data before passing to the next node
  - Aggregating or modifying event data dynamically

### 5. WebhookReceiver Node

- **Purpose**: Event ingestion from external webhooks
- **Features**:
  - Configurable endpoints for different webhook sources
  - Payload parsing and validation
  - Security features like secret tokens and request validation
- **Use Cases**:
  - Integrating external triggers into workflows
  - Receiving real-time event data from third-party services

### 6. Notification Node

- **Purpose**: Automated notifications through various channels
- **Features**:
  - Supports multiple channels (Email, Slack, SMS, etc.)
  - Customizable message templates based on event data
  - Scheduling options for delayed notifications
- **Use Cases**:
  - Sending alerts based on workflow events
  - Notifying teams of critical incidents or updates

### 7. Logger Node

- **Purpose**: Detailed logging for monitoring and debugging
- **Features**:
  - Multiple log levels (INFO, DEBUG, ERROR)
  - Log storage options (console, files, external services)
  - Structured logging in JSON or plain text formats
- **Use Cases**:
  - Capturing and storing workflow events for analysis
  - Monitoring workflow performance and troubleshooting issues

### 8. Retry Node

- **Purpose**: Resilient workflow with retry mechanisms
- **Features**:
  - Configurable retry counts and backoff strategies (exponential, fixed)
  - Conditional retries based on error types or conditions
  - Emits error events after max retries are exhausted
- **Use Cases**:
  - Retrying failed API requests or database operations
  - Ensuring reliable event processing in workflows

### 9. BatchProcessor Node

- **Purpose**: Batch processing of accumulated events
- **Features**:
  - Configurable batch sizes and time intervals
  - Aggregation functions for batch data
  - Emits processed batch events
- **Use Cases**:
  - Processing multiple events together for efficiency
  - Summarizing or aggregating data before passing to the next node

### 10. Cache Node

- **Purpose**: Performance optimization through data caching
- **Features**:
  - Configurable cache durations and invalidation rules
  - Customizable cache keys for different data types
  - Serves cached data to reduce redundant processing
- **Use Cases**:
  - Caching frequent API responses
  - Reducing load on external services by serving cached data

## Node Communication

Nodes communicate through WebSocket connections using the standardized event system. Here's how different node types handle events:

### LLM Node

- **Input**: Receives text events as prompts
- **Output**: Emits chunk events during generation, complete event when done
- **Error**: Emits error events for failed requests

### Leaky Bucket Node

- **Input**: Receives any event type
- **Output**: Rate-limited chunk events
- **State**: Maintains internal buffer of events

### Spawner Node

- **Input**: Spawn requests with configuration
- **Output**: Spawn events for created nodes
- **Cleanup**: Remove events when nodes expire

### Timer Node

- **Output**: Tick events at specified intervals
- **State**: Active/Inactive status events

### Condition Node

- **Input**: Receives events to evaluate conditions
- **Output**: Emits `conditionTrue` or `conditionFalse` events based on evaluation
- **Error**: Emits error events for failed condition evaluations

### Database Node

- **Input**: Receives events with database operation details
- **Output**: Emits success or error events based on operation results

### APIRequest Node

- **Input**: Receives events with API request details
- **Output**: Emits API response or error events based on request outcomes

### Transform Node

- **Input**: Receives events to transform
- **Output**: Emits transformed data or error events if transformation fails

### WebhookReceiver Node

- **Input**: Receives external webhook events
- **Output**: Emits ingested webhook data as events

### Notification Node

- **Input**: Receives events to trigger notifications
- **Output**: Emits success or error events based on notification delivery

### Logger Node

- **Input**: Receives all events for logging purposes
- **Output**: Typically does not emit further events unless logging fails

### Retry Node

- **Input**: Receives error events to trigger retries
- **Output**: Emits retry attempts or final error events after max retries

### BatchProcessor Node

- **Input**: Receives events to accumulate in batches
- **Output**: Emits processed batch events based on configuration

### Cache Node

- **Input**: Receives events to cache or retrieve from cache
- **Output**: Emits cached data or passes through uncached data

## Event Flow Control

Events can be:

1. **Chained**: Using parent/child relationships
2. **Sequenced**: Using sequence numbers
3. **Rate-limited**: Through LeakyBucket nodes
4. **Transformed**: Nodes can modify event data
5. **Filtered**: Nodes can selectively process events

## Debugging

Each node supports debug mode that logs:

1. Incoming events
2. Outgoing events
3. State changes
4. Error conditions

Debug output includes:

- Event IDs for tracing
- Timestamps
- Parent/child relationships
- Sequence information

## Testing

The event system can be tested using the provided test suite:

```bash
# Run all tests
npm test

# Run specific node tests
npm run test:leaky-bucket
npm run test:llm
npm run test:spawner

# Watch mode for development
npm run test:watch
```

Tests verify:

1. Event structure compliance
2. Proper event flow
3. Rate limiting behavior
4. Error handling
5. Event sequencing
6. Parent/child relationships

## Best Practices

1. **Event Creation**:
   - Always use helper functions
   - Include relevant metadata
   - Set parent IDs for related events

2. **Event Handling**:
   - Validate event structure
   - Check event types
   - Handle errors gracefully

3. **Flow Control**:
   - Use LeakyBucket for rate limiting
   - Maintain event ordering
   - Clean up completed sequences

4. **Debugging**:
   - Enable debug mode selectively
   - Monitor event flow
   - Track event relationships

## Future Enhancements

1. **Event Persistence**:
   - Event storage
   - Event replay
   - Flow analysis

2. **Advanced Flow Control**:
   - Event prioritization
   - Conditional routing
   - Dynamic rate limiting

3. **Monitoring**:
   - Event metrics
   - Flow visualization
   - Performance analysis 

## Innovative Use Cases

LLM Graph 7's event-driven, node-based architecture offers a versatile foundation for various applications. Below is a comprehensive list of practical applications:

### Content & Media

1. Generate daily social media posts with automated A/B testing of engagement
2. Create automated podcast show notes from audio transcripts
3. Build a content repurposing pipeline (blog → social → newsletter)
4. Generate and schedule localized content for different time zones
5. Create automated video script outlines from trending topics
6. Build a real-time news aggregator and summarizer with sentiment analysis
7. Generate product descriptions from technical specifications
8. Create automated documentation updates from git commits
9. Build a multi-source research paper summarization system
10. Generate SEO-optimized meta descriptions for website content

### Development & Testing

11. Create automated code review assistant with best practice suggestions
12. Generate unit tests from code changes
13. Build a bug report analyzer and priority sorter
14. Create automated API documentation from endpoint usage
15. Generate code examples from documentation
16. Build a commit message quality checker and improver
17. Create automated release notes from git history
18. Generate code refactoring suggestions based on static analysis
19. Build a dependency update impact analyzer
20. Create automated PR description generator

### Business & Analytics

21. Build a customer feedback categorization and response system
22. Create automated competitive analysis reports from web scraping
23. Generate sales email templates based on prospect data
24. Build a real-time market sentiment analyzer
25. Create automated financial report summaries
26. Generate meeting minutes from audio transcripts
27. Build a customer support ticket prioritization system
28. Create automated expense report categorization
29. Generate performance review summaries from multiple data sources
30. Build a lead scoring system based on multiple criteria

### Education & Learning

31. Create personalized study guides from course materials
32. Build an automated quiz generator from educational content
33. Generate simplified explanations of complex topics
34. Create language learning exercises from real-world content
35. Build a homework helper that explains step-by-step solutions
36. Generate flashcards from lecture notes
37. Create educational content adaptations for different age groups
38. Build a concept relationship mapper for study topics
39. Generate practice problems with varying difficulty levels
40. Create automated progress reports for students

### Research & Analysis

41. Build a research paper recommendation system based on reading history
42. Create automated literature review summaries
43. Generate research methodology comparisons
44. Build a citation network analyzer
45. Create automated experiment log summarizers
46. Generate research proposal drafts from project notes
47. Build a data analysis report generator
48. Create automated patent analysis reports
49. Generate research gap identification reports
50. Build a cross-discipline research connector

### Personal Productivity

51. Create an email summarizer and priority sorter
52. Build a personal knowledge base curator
53. Generate weekly productivity reports from activity logs
54. Create automated task prioritization based on multiple inputs
55. Build a meeting scheduler with context-aware scheduling
56. Generate personal development plans from goal tracking
57. Create automated diary entries from daily activities
58. Build a habit tracking analyzer with improvement suggestions
59. Generate personalized reading recommendations
60. Create automated time-blocking schedules

### Data Processing

61. Build a data cleaning and normalization pipeline
62. Create automated data quality reports
63. Generate data transformation rules from examples
64. Build a multi-source data aggregator with custom merging
65. Create automated data validation checks
66. Generate data visualization suggestions
67. Build a data anomaly detection system
68. Create automated dataset documentation
69. Generate data sampling strategies
70. Build a data versioning system with change summaries

### AI/ML Operations

71. Create automated model performance reports
72. Build a training data quality analyzer
73. Generate model behavior comparison reports
74. Create automated model documentation
75. Build a model prediction explainer
76. Generate feature importance analysis
77. Create automated model retraining triggers
78. Build a model version comparison system
79. Generate model bias analysis reports
80. Create automated model deployment validation

### Integration & Automation

81. Build a multi-API workflow orchestrator
82. Create automated API integration tests
83. Generate webhook handling logic
84. Build a service health monitoring system
85. Create automated system integration reports
86. Generate API usage optimization suggestions
87. Build a service dependency mapper
88. Create automated failover procedures
89. Generate integration documentation
90. Build a cross-service error correlation system

### Monitoring & Debugging

91. Create automated error pattern analysis
92. Build a system performance bottleneck detector
93. Generate debug log summaries
94. Create automated incident reports
95. Build a real-time system health dashboard
96. Generate performance optimization suggestions
97. Create automated recovery procedures
98. Build a user impact analyzer for system issues
99. Generate system stability reports
100. Create automated alert correlation analysis

## Recommended New Node Types

Based on the extensive list of scenarios and current capabilities, introducing the following new node types can significantly expand LLM Graph 7's functionality:

### 1. Condition Node

- **Purpose**: Flow control and event filtering based on specified conditions.
- **Features**:
  - **JavaScript Condition Evaluation**: Executes custom JavaScript conditions to determine event flow.
  - **Branching**: Directs events to different nodes based on condition outcomes (`true` or `false`).
  - **Looping Mode**: Option to re-trigger events for iterative processing.

### 2. Database Node

- **Purpose**: Data persistence and database interactions.
- **Features**:
  - **Multiple Database Support**: Compatible with SQL, NoSQL, and other database types.
  - **Parameterized Queries**: Executes secure, parameterized queries to prevent injection attacks.
  - **Connection Management**: Handles database connections efficiently, including pooling and reconnections.

### 3. APIRequest Node

- **Purpose**: External API integration.
- **Features**:
  - **HTTP Methods Support**: Supports GET, POST, PUT, DELETE, etc.
  - **Authentication Handling**: Manages API keys, OAuth tokens, and other authentication mechanisms.
  - **Response Parsing**: Automatically parses JSON, XML, or other response formats.

### 4. Transform Node

- **Purpose**: Data transformation and processing.
- **Features**:
  - **Custom Transformation Logic**: Allows users to define JavaScript functions for data manipulation.
  - **Chainable Transformations**: Supports multiple transformation steps in a single node.
  - **Error Handling**: Catches and emits errors during data transformation.

### 5. WebhookReceiver Node

- **Purpose**: Event ingestion from external webhooks.
- **Features**:
  - **URL Configuration**: Sets up unique endpoints for different webhook sources.
  - **Payload Parsing**: Automatically parses incoming webhook payloads.
  - **Security**: Supports secret tokens and validation to ensure secure event ingestion.

### 6. Notification Node

- **Purpose**: Automated notifications through various channels.
- **Features**:
  - **Multiple Channels Support**: Integrates with email services, Slack APIs, SMS gateways, etc.
  - **Customizable Messages**: Allows dynamic message content based on event data.
  - **Scheduling**: Option to delay notifications based on conditions or schedules.

### 7. Logger Node

- **Purpose**: Detailed logging for monitoring and debugging purposes.
- **Features**:
  - **Multiple Log Levels**: Supports INFO, DEBUG, ERROR, etc.
  - **Log Storage Options**: Can log to console, files, or external logging services.
  - **Structured Logging**: Formats logs in JSON or other structured formats for easy parsing.

### 8. Retry Node

- **Purpose**: Resilient workflow with retry mechanisms.
- **Features**:
  - **Configurable Retry Count**: Sets how many times to retry an event upon failure.
  - **Backoff Strategies**: Implements exponential or fixed delays between retries.
  - **Conditional Retries**: Retries based on specific error types or conditions.

### 9. BatchProcessor Node

- **Purpose**: Batch processing of accumulated events.
- **Features**:
  - **Batch Size Configuration**: Sets the number of events to accumulate before processing.
  - **Time-Based Triggers**: Processes batches after a certain time interval even if batch size is not met.
  - **Aggregation Functions**: Supports aggregating or summarizing data within the batch.

### 10. Cache Node

- **Purpose**: Performance optimization through data caching.
- **Features**:
  - **Configurable Cache Duration**: Sets how long data remains cached.
  - **Cache Invalidation**: Defines rules for invalidating or updating cached data.
  - **Cache Key Management**: Supports customizable keys for different data types.

## Node Communication

Nodes communicate through WebSocket connections using the standardized event system. Here's how different node types handle events:

### LLM Node

- **Input**: Receives text events as prompts
- **Output**: Emits chunk events during generation, complete event when done
- **Error**: Emits error events for failed requests

### Leaky Bucket Node

- **Input**: Receives any event type
- **Output**: Rate-limited chunk events
- **State**: Maintains internal buffer of events

### Spawner Node

- **Input**: Spawn requests with configuration
- **Output**: Spawn events for created nodes
- **Cleanup**: Remove events when nodes expire

### Timer Node

- **Output**: Tick events at specified intervals
- **State**: Active/Inactive status events

### Condition Node

- **Input**: Receives events to evaluate conditions
- **Output**: Emits `conditionTrue` or `conditionFalse` events based on evaluation
- **Error**: Emits error events for failed condition evaluations

### Database Node

- **Input**: Receives events with database operation details
- **Output**: Emits success or error events based on operation results

### APIRequest Node

- **Input**: Receives events with API request details
- **Output**: Emits API response or error events based on request outcomes

### Transform Node

- **Input**: Receives events to transform
- **Output**: Emits transformed data or error events if transformation fails

### WebhookReceiver Node

- **Input**: Receives external webhook events
- **Output**: Emits ingested webhook data as events

### Notification Node

- **Input**: Receives events to trigger notifications
- **Output**: Emits success or error events based on notification delivery

### Logger Node

- **Input**: Receives all events for logging purposes
- **Output**: Typically does not emit further events unless logging fails

### Retry Node

- **Input**: Receives error events to trigger retries
- **Output**: Emits retry attempts or final error events after max retries

### BatchProcessor Node

- **Input**: Receives events to accumulate in batches
- **Output**: Emits processed batch events based on configuration

### Cache Node

- **Input**: Receives events to cache or retrieve from cache
- **Output**: Emits cached data or passes through uncached data

## Event Flow Control

Events can be:

1. **Chained**: Using parent/child relationships
2. **Sequenced**: Using sequence numbers
3. **Rate-limited**: Through LeakyBucket nodes
4. **Transformed**: Nodes can modify event data
5. **Filtered**: Nodes can selectively process events

## Debugging

Each node supports debug mode that logs:

1. Incoming events
2. Outgoing events
3. State changes
4. Error conditions

Debug output includes:

- Event IDs for tracing
- Timestamps
- Parent/child relationships
- Sequence information

## Testing

The event system can be tested using the provided test suite:

```bash
# Run all tests
npm test

# Run specific node tests
npm run test:leaky-bucket
npm run test:llm
npm run test:spawner

# Watch mode for development
npm run test:watch
```

Tests verify:

1. Event structure compliance
2. Proper event flow
3. Rate limiting behavior
4. Error handling
5. Event sequencing
6. Parent/child relationships

## Best Practices

1. **Event Creation**:
   - Always use helper functions
   - Include relevant metadata
   - Set parent IDs for related events

2. **Event Handling**:
   - Validate event structure
   - Check event types
   - Handle errors gracefully

3. **Flow Control**:
   - Use LeakyBucket for rate limiting
   - Maintain event ordering
   - Clean up completed sequences

4. **Debugging**:
   - Enable debug mode selectively
   - Monitor event flow
   - Track event relationships

## Future Enhancements

1. **Event Persistence**:
   - Event storage
   - Event replay
   - Flow analysis

2. **Advanced Flow Control**:
   - Event prioritization
   - Conditional routing
   - Dynamic rate limiting

3. **Monitoring**:
   - Event metrics
   - Flow visualization
   - Performance analysis 

## Summary of Recent Changes

1. **Fixed Streaming Functionality in LLM Node**:
   - Updated server to use `ollama.generate` with `stream: true`
   - Properly handles streaming chunks through WebSocket
   - Fixed connection handling between nodes

2. **Added Debug Mode to Nodes**:
   - Added debug toggle button to each node
   - Implemented node-specific debug output panels
   - Shows progress indicators (|) and status ([complete], [error])
   - Removed console spam in favor of in-node debugging

3. **Updated Documentation**:
   - Documented current node implementations (LLM, Timer, Output)
   - Added planned node types section
   - Updated setup instructions for Ollama integration
   - Added debug mode documentation

4. **Implemented New Node Types**:
   - **LeakyBucket Node** for token accumulation and rate limiting
   - **Merger Node** for combining and summarizing content
   - Added node registration in `App.jsx`
   - Updated `Sidebar` with new node types

## Current Issues

1. **Content Flow Problem**:
   - LeakyBucket nodes not receiving content from spawned nodes
   - Connections are being made (logged in `App.jsx`)
   - Need to debug WebSocket message handling
   - Verify spawned node output emission

2. **Ollama Request Cleanup**:
   - Problem: Ollama requests continue generating after client disconnection
   - Added AbortController but cleanup not working as expected
   - Added logging to track abort calls and request states

3. **LeakyBucket Tests**:
   - Some tests failing related to event buffering and leaking mechanisms
   - Need to review and fix event handling

## Next Steps

1. **Fix Content Flow from Spawned Nodes to LeakyBucket**:
   - Debug WebSocket message handling
   - Ensure proper event emission from spawned nodes

2. **Complete Testing of Merger Node Functionality**:
   - Develop comprehensive tests for merging logic
   - Ensure accurate summarization of combined content

3. **Implement Remaining Planned Nodes**:
   - **Condition Node**: For flow control and event filtering
   - **Memory Node**: For maintaining context and state
   - **Data Source Node**: For external data integration

4. **Debug Ollama Request Cleanup**:
   - Verify AbortController implementation
   - Check Ollama library's cancellation support
   - Consider alternative cleanup approaches

5. **Fix LeakyBucket Tests**:
   - Review event buffering logic
   - Fix event leaking mechanisms
   - Update tests for new event structure

6. **Continue Standardizing Events**:
   - Implement in remaining nodes
   - Add more test coverage
   - Update documentation as needed

## Project Structure

```
llm-graph7/
├── src/
│   ├── components/
│   │   ├── nodes/
│   │   │   ├── LLMNode.jsx
│   │   │   ├── OutputNode.jsx
│   │   │   ├── TimerNode.jsx
│   │   │   ├── LeakyBucketNode.jsx
│   │   │   └── MergerNode.jsx
│   │   └── Sidebar.jsx
│   └── App.jsx
└── server/
     └── server.js
```

The system has been expanded with new node types for token accumulation and content merging but requires debugging of the content flow between spawned nodes and LeakyBucket nodes. The core streaming LLM capabilities and debug features remain functional.

## LLM Graph Project Context

This document provides context about the LLM Graph project, its current state, and recent changes.

### Recent Changes and Current State

1. **Standardized Event System Implementation**
   - Created base event interface with fields: `id`, `type`, `timestamp`, `source`, `metadata`, `content`
   - Implemented helper functions for creating events (`chunk`, `completion`, `error`)
   - Updated nodes to use standardized events (LLM, LeakyBucket, Spawner)

2. **Test Framework Setup**
   - Installed Mocha and Chai for testing
   - Created test files for LeakyBucket, LLM, and Spawner nodes
   - Added test scripts to `package.json`

3. **Documentation**
   - Added comprehensive event system documentation to `README.md`
   - Documented event structure, types, flows, and best practices
   - Included examples and future enhancements

### Current Issues

1. **Ollama Request Cleanup**
   - Problem: Ollama requests continue generating after client disconnection
   - Added AbortController but cleanup not working as expected
   - Added logging to track abort calls and request states

2. **LeakyBucket Tests**
   - Some tests failing related to event buffering and leaking mechanisms
   - Need to review and fix event handling

### Next Steps

1. **Debug Ollama Request Cleanup**
   - Verify AbortController implementation
   - Check Ollama library's cancellation support
   - Consider alternative cleanup approaches

2. **Fix LeakyBucket Tests**
   - Review event buffering logic
   - Fix event leaking mechanisms
   - Update tests for new event structure

3. **Continue Standardizing Events**
   - Implement in remaining nodes
   - Add more test coverage
   - Update documentation as needed

### Technical Details

1. **Event Structure**

   ```typescript
   interface Event {
     id: string;          // Unique event ID
     type: string;        // Event type
     source: string;      // Source node ID
     data: any;           // Payload
     meta?: {             // Optional metadata
       parent?: string;   // Parent event ID
       seq?: number;      // Sequence number
       done?: boolean;    // Completion marker
     }
   }
   ```

2. **Main Event Types**

   - `chunk`: Partial content from streaming
   - `complete`: Sequence completion
   - `error`: Error events
   - `spawn`: Node creation
   - `remove`: Node removal
   - `tick`: Timer events
   - `state`: State changes

3. **Active Development Files**

   - `llm-graph7/server/server.js`
   - `llm-graph7/test/nodes/*.test.js`
   - `llm-graph7/README.md`

## Best Practices

1. **Event Creation**:
   - Always use helper functions
   - Include relevant metadata
   - Set parent IDs for related events

2. **Event Handling**:
   - Validate event structure
   - Check event types
   - Handle errors gracefully

3. **Flow Control**:
   - Use LeakyBucket for rate limiting
   - Maintain event ordering
   - Clean up completed sequences

4. **Debugging**:
   - Enable debug mode selectively
   - Monitor event flow
   - Track event relationships

## Future Enhancements

1. **Event Persistence**:
   - Event storage
   - Event replay
   - Flow analysis

2. **Advanced Flow Control**:
   - Event prioritization
   - Conditional routing
   - Dynamic rate limiting

3. **Monitoring**:
   - Event metrics
   - Flow visualization
   - Performance analysis 

## Additional Resources

- [React Flow Documentation](https://reactflow.dev/)
- [Express.js Documentation](https://expressjs.com/)
- [Fetch API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)

## License

This project is licensed under the MIT License.

## Acknowledgments

- Inspired by the robust event-driven architecture and modular design principles.
- Leveraging community feedback and practical use case scenarios to inform enhancements.

## Contact

For contributions and support, please open issues or submit pull requests through the project's GitHub repository.

# Happy Coding!
```
