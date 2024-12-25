import express from 'express';
import { WebSocketServer, WebSocket } from 'ws';
import { createServer } from 'http';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import cors from 'cors';
import { Graph } from '../src/graph/graph.js';
import { Ollama } from 'ollama';

// Log levels and configuration
const LOG_LEVELS = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3
};

const CURRENT_LOG_LEVEL = LOG_LEVELS.INFO; // Change this to DEBUG for more verbose logging

function log(level, ...args) {
  if (level >= CURRENT_LOG_LEVEL) {
    const prefix = Object.keys(LOG_LEVELS).find(key => LOG_LEVELS[key] === level);
    console.log(`[${prefix}]`, ...args);
  }
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const httpServer = createServer(app);
const PORT = 3000;

// Initialize Ollama host
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
log(LOG_LEVELS.INFO, `Ollama host configured at: ${OLLAMA_HOST}`);

// Remove global ollama client - we'll create per-node instances

// Enable CORS for development
app.use(cors({
  origin: 'http://localhost:5173',
  methods: ['GET', 'POST'],
  credentials: true,
}));
app.use(express.json());

// Serve static files
app.use(express.static(join(__dirname, '..')));

// Initialize WebSocket Server with path '/ws'
const wss = new WebSocketServer({ server: httpServer, path: '/ws' });

const graph = new Graph();

function broadcast(message) {
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(message));
    }
  });
}

// Handle graph events
graph.on('nodeAdded', (node) => {
  broadcast({ type: 'nodeAdded', node: serializeNode(node) });
});

graph.on('connectionAdded', (sourceNodeId, targetNodeId) => {
  broadcast({ type: 'connectionAdded', sourceNodeId, targetNodeId });
});

graph.on('nodeRemoved', (nodeId) => {
  broadcast({ type: 'nodeRemoved', nodeId });
});

graph.on('connectionRemoved', (sourceNodeId, targetNodeId) => {
  broadcast({ type: 'connectionRemoved', sourceNodeId, targetNodeId });
});

const activeTimers = new Map();
const spawnedNodes = new Map(); // Track spawned nodes and their timeouts
const ollamaProcesses = new Map(); // Track Ollama processes and their state

function cleanupSpawnedNode(nodeId) {
  const spawnedNode = spawnedNodes.get(nodeId);
  if (spawnedNode) {
    log(LOG_LEVELS.INFO, `[Spawn] Cleaning up spawned node ${nodeId}`);
    clearTimeout(spawnedNode.cleanupTimeout);
    if (spawnedNode.type === 'timer' && activeTimers.has(nodeId)) {
      clearInterval(activeTimers.get(nodeId));
      activeTimers.delete(nodeId);
    }
    spawnedNodes.delete(nodeId);
    
    // Broadcast node removal for GUI
    broadcast({
      type: 'nodeRemoved',
      nodeId: nodeId,
      parentNodeId: spawnedNode.parentId,
      timestamp: Date.now(),
      formattedTime: new Date().toLocaleTimeString()
    });
  }
}

async function handleSpawnNode(data) {
  const { nodeId: parentId, nodeType, timeout, promptData } = data;
  log(LOG_LEVELS.INFO, '[Spawn] Handling spawn request:', { parentId, nodeType, timeout, promptData });
  
  const spawnedNodeId = `spawned_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  log(LOG_LEVELS.DEBUG, '[Spawn] Generated spawned node ID:', spawnedNodeId);

  // Create spawned node tracking
  spawnedNodes.set(spawnedNodeId, {
    type: nodeType,
    parentId,
    promptData,
    cleanupTimeout: setTimeout(() => {
      log(LOG_LEVELS.INFO, `[Spawn] Cleanup timeout triggered for node: ${spawnedNodeId}`);
      cleanupSpawnedNode(spawnedNodeId);
    }, timeout)
  });

  // Calculate better position offset from parent
  const spawnIndex = Array.from(spawnedNodes.values()).filter(n => n.parentId === parentId).length - 1;
  const offsetX = 300; // Increased horizontal spacing
  const offsetY = spawnIndex * 200 - 200; // Increased vertical spacing
  log(LOG_LEVELS.DEBUG, '[Spawn] Calculated position offset:', { offsetX, offsetY, spawnIndex });

  const nodeData = {
    id: spawnedNodeId,
    type: nodeType,
    position: { x: offsetX, y: offsetY },
    data: { 
      label: promptData ? `${promptData.role}` : `Spawned ${nodeType} Node`,
      socket: null, // Will be set by client
      promptData // Pass prompt data to node for display
    },
    parentId: parentId,
    style: { // Add custom styling
      width: 250,
      border: promptData ? '2px solid #2196F3' : '1px solid #ddd'
    }
  };

  // Broadcast node creation with position
  const spawnMessage = {
    type: 'nodeSpawned',
    spawnedNodeId,
    parentNodeId: parentId,
    nodeType,
    promptData,
    nodeData,
    timestamp: Date.now(),
    formattedTime: new Date().toLocaleTimeString()
  };
  
  log(LOG_LEVELS.DEBUG, '[Spawn] Broadcasting spawn message:', spawnMessage);
  broadcast(spawnMessage);

  // If it's an LLM node, automatically send the analysis prompt
  if (nodeType === 'llm' && promptData) {
    log(LOG_LEVELS.INFO, `[Spawn] Scheduling LLM request for spawned node: ${spawnedNodeId}`);
    setTimeout(() => {
      log(LOG_LEVELS.INFO, `[Spawn] Sending LLM request for spawned node: ${spawnedNodeId}`);
      handleLLMRequest(spawnedNodeId, promptData.prompt, 'smollm:latest');
    }, 500); // Small delay to ensure node is ready
  }

  return spawnedNodeId;
}

// Function to get available Ollama models
async function getAvailableModels() {
  try {
    // Create temporary client for model listing
    const modelClient = new Ollama({ host: OLLAMA_HOST });
    const response = await modelClient.list();
    log(LOG_LEVELS.DEBUG, 'Available models:', response);
    return response.models.map(model => ({
      name: model.name,
      size: model.size,
      modified: model.modified_at,
      digest: model.digest
    }));
  } catch (error) {
    log(LOG_LEVELS.ERROR, 'Error fetching models:', error);
    return [];
  }
}

// Create a new Ollama instance for each node
function createNodeOllamaClient(nodeId) {
  log(LOG_LEVELS.DEBUG, `Creating new Ollama client for node ${nodeId}`);
  const client = new Ollama({
    host: OLLAMA_HOST
  });
  return client;
}

async function handleLLMRequest(nodeId, prompt, model) {
  try {
    // Create a dedicated Ollama client for this node
    const nodeOllama = createNodeOllamaClient(nodeId);
    
    // Track the request state
    const processState = {
      ollama: nodeOllama,
      isProcessing: true,
      lastChunkTime: Date.now(),
      nodeId
    };
    ollamaProcesses.set(nodeId, processState);

    log(LOG_LEVELS.INFO, `[Ollama] Starting request for node ${nodeId} with model ${model}`);

    // Set up a watchdog timer to detect stalled requests
    const watchdog = setInterval(() => {
      if (processState.isProcessing && Date.now() - processState.lastChunkTime > 5000) {
        log(LOG_LEVELS.WARN, `[Ollama] Request for node ${nodeId} appears stalled, forcing abort`);
        clearInterval(watchdog);
        forceAbortOllama(nodeId);
      }
    }, 1000);

    try {
      const stream = await nodeOllama.generate({
        model: model,
        prompt: prompt,
        stream: true,
        options: {
          num_predict: -1  // Ensure we can abort mid-generation
        }
      });

      for await (const chunk of stream) {
        // Check if processing was aborted
        if (!processState.isProcessing) {
          log(LOG_LEVELS.INFO, `[Ollama] Request aborted for node ${nodeId}, stopping chunk processing`);
          break;
        }

        if (chunk.response) {
          processState.lastChunkTime = Date.now();
          const message = {
            type: 'llmResponseChunk',
            nodeId: nodeId,
            chunk: chunk.response,
            timestamp: Date.now(),
            formattedTime: new Date().toLocaleTimeString()
          };
          broadcast(message);
          process.stdout.write('|');
        }
      }

      if (processState.isProcessing) {
        broadcast({
          type: 'llmResponseComplete',
          nodeId: nodeId,
          timestamp: Date.now(),
          formattedTime: new Date().toLocaleTimeString()
        });
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        log(LOG_LEVELS.INFO, `[Ollama] Request aborted for node ${nodeId}`);
        broadcast({
          type: 'llmAborted',
          nodeId: nodeId,
          timestamp: Date.now(),
          formattedTime: new Date().toLocaleTimeString()
        });
      } else {
        throw error;  // Re-throw non-abort errors
      }
    } finally {
      clearInterval(watchdog);
      processState.isProcessing = false;
      ollamaProcesses.delete(nodeId);
    }

  } catch (error) {
    log(LOG_LEVELS.ERROR, `[Ollama] Error in LLM request for node ${nodeId}:`, error.message);
    broadcast({
      type: 'llmError',
      nodeId: nodeId,
      error: error.message,
      timestamp: Date.now(),
      formattedTime: new Date().toLocaleTimeString()
    });
  }
}

// Force abort an Ollama request with cleanup
async function forceAbortOllama(nodeId) {
  log(LOG_LEVELS.INFO, `[Ollama] Force aborting request for node ${nodeId}`);
  
  const processState = ollamaProcesses.get(nodeId);
  
  if (processState) {
    processState.isProcessing = false;
    try {
      // Use the node's Ollama client to abort
      log(LOG_LEVELS.DEBUG, `[Ollama] Calling abort() for node ${nodeId}`);
      await processState.ollama.abort();
      log(LOG_LEVELS.INFO, `[Ollama] Successfully aborted request for node ${nodeId}`);
    } catch (error) {
      log(LOG_LEVELS.ERROR, `[Ollama] Error aborting request for node ${nodeId}:`, error);
      log(LOG_LEVELS.INFO, `[Ollama] Forcing cleanup for node ${nodeId} despite abort error`);
    }
  } else {
    log(LOG_LEVELS.WARN, `[Ollama] No process state found for node ${nodeId}, skipping abort`);
  }

  // Clean up tracking
  ollamaProcesses.delete(nodeId);

  // Send abort notification
  broadcast({
    type: 'llmAborted',
    nodeId: nodeId,
    timestamp: Date.now(),
    formattedTime: new Date().toLocaleTimeString()
  });
}

// WebSocket connection handling
wss.on('connection', async (ws) => {
  console.log('Client connected');

  // Send available models to client on connection
  try {
    const models = await getAvailableModels();
    ws.send(JSON.stringify({
      type: 'availableModels',
      models: models
    }));
  } catch (error) {
    console.error('Error sending models to client:', error);
  }

  ws.on('message', async (message) => {
    let parsedData;
    try {
      parsedData = JSON.parse(message);
      
      // Only log non-content messages at INFO level, content messages at DEBUG
      if (parsedData.type === 'content') {
        log(LOG_LEVELS.DEBUG, 'Content message:', {
          nodeId: parsedData.nodeId,
          contentLength: parsedData.content?.length || 0,
          isComplete: parsedData.metadata?.isComplete
        });
      } else {
        log(LOG_LEVELS.INFO, 'Received message:', parsedData.type);
        log(LOG_LEVELS.DEBUG, 'Message details:', JSON.stringify(parsedData, null, 2));
      }
      
      if (parsedData.type === 'startTimer') {
        log(LOG_LEVELS.INFO, `[Timer] Starting timer for node ${parsedData.nodeId} with interval ${parsedData.interval}ms`);
        
        // Clear existing timer if any
        if (activeTimers.has(parsedData.nodeId)) {
          log(LOG_LEVELS.DEBUG, `[Timer] Clearing existing timer for node ${parsedData.nodeId}`);
          clearInterval(activeTimers.get(parsedData.nodeId));
          activeTimers.delete(parsedData.nodeId);
        }
        
        // Create new timer
        log(LOG_LEVELS.DEBUG, `[Timer] Creating new timer interval`);
        const timerId = setInterval(() => {
          log(LOG_LEVELS.DEBUG, `[Timer] Tick for node ${parsedData.nodeId}`);
          const timerEvent = {
            type: 'timerTick',
            nodeId: parsedData.nodeId,
            timestamp: Date.now(),
            formattedTime: new Date().toLocaleTimeString()
          };
          log(LOG_LEVELS.DEBUG, `[Timer] Broadcasting tick event:`, JSON.stringify(timerEvent));
          broadcast(timerEvent);
          process.stdout.write('T');
        }, parsedData.interval);
        
        // Store timer reference
        activeTimers.set(parsedData.nodeId, timerId);
        log(LOG_LEVELS.DEBUG, `[Timer] Timer started for node ${parsedData.nodeId}, active timers:`, Array.from(activeTimers.keys()));
        
        // Send confirmation back to client
        ws.send(JSON.stringify({
          type: 'timerStarted',
          nodeId: parsedData.nodeId,
          timestamp: Date.now(),
          formattedTime: new Date().toLocaleTimeString()
        }));
        
      } else if (parsedData.type === 'stopTimer') {
        log(LOG_LEVELS.INFO, `[Timer] Stopping timer for node ${parsedData.nodeId}`);
        if (activeTimers.has(parsedData.nodeId)) {
          clearInterval(activeTimers.get(parsedData.nodeId));
          activeTimers.delete(parsedData.nodeId);
          log(LOG_LEVELS.INFO, `[Timer] Timer stopped for node ${parsedData.nodeId}, remaining timers:`, Array.from(activeTimers.keys()));
          
          // Send confirmation back to client
          ws.send(JSON.stringify({
            type: 'timerStopped',
            nodeId: parsedData.nodeId,
            timestamp: Date.now(),
            formattedTime: new Date().toLocaleTimeString()
          }));
        } else {
          log(LOG_LEVELS.INFO, `[Timer] No active timer found for node ${parsedData.nodeId}`);
        }
      } else if (parsedData.type === 'llmRequest') {
        log(LOG_LEVELS.INFO, 'Processing LLM request for node:', parsedData.nodeId);
        await handleLLMRequest(parsedData.nodeId, parsedData.prompt, parsedData.model);
      } else if (parsedData.type === 'getModels') {
        log(LOG_LEVELS.INFO, 'Fetching available models');
        const models = await getAvailableModels();
        ws.send(JSON.stringify({
          type: 'availableModels',
          models: models
        }));
      } else if (parsedData.type === 'spawnNode') {
        log(LOG_LEVELS.INFO, 'Spawning node:', parsedData.nodeType, 'parent:', parsedData.nodeId);
        const spawnedId = await handleSpawnNode(parsedData);
        log(LOG_LEVELS.INFO, 'Spawned node created:', spawnedId);
      }
      
    } catch (error) {
      log(LOG_LEVELS.ERROR, 'Error processing message:', error);
      if (parsedData?.nodeId) {
        broadcast({
          type: 'nodeError',
          nodeId: parsedData.nodeId,
          error: error.message,
          timestamp: Date.now(),
          formattedTime: new Date().toLocaleTimeString()
        });
      }
    }
  });

  ws.on('close', () => {
    log(LOG_LEVELS.INFO, 'Client disconnected');
    
    // Log current state
    log(LOG_LEVELS.INFO, `[Ollama] Active processes before cleanup: ${Array.from(ollamaProcesses.keys()).join(', ')}`);
    
    // Cleanup any timers associated with this connection
    for (const [nodeId, timerId] of activeTimers.entries()) {
      log(LOG_LEVELS.INFO, `[Timer] Cleaning up timer for node ${nodeId} on disconnect`);
      clearInterval(timerId);
    }
    activeTimers.clear();

    // Force abort any active Ollama requests
    const abortPromises = [];
    for (const [nodeId] of ollamaProcesses.entries()) {
      log(LOG_LEVELS.INFO, `[Ollama] Force aborting request for node ${nodeId} on disconnect`);
      abortPromises.push(forceAbortOllama(nodeId));
    }

    // Wait for all aborts to complete
    Promise.all(abortPromises).then(() => {
      log(LOG_LEVELS.INFO, '[Ollama] All abort operations completed');
      log(LOG_LEVELS.DEBUG, `[Ollama] Remaining processes: ${Array.from(ollamaProcesses.keys()).join(', ')}`);
    }).catch(error => {
      log(LOG_LEVELS.ERROR, '[Ollama] Error during abort cleanup:', error);
    });
  });
});

// Handle process termination
process.on('SIGTERM', cleanup);
process.on('SIGINT', cleanup);

function cleanup() {
  console.log('\nShutting down server...');
  
  // Force abort all active Ollama requests
  for (const [nodeId] of ollamaProcesses.entries()) {
    forceAbortOllama(nodeId);
  }

  // Close all WebSocket connections
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.close(1000, 'Server shutting down');
    }
  });

  // Close the server
  httpServer.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
}

// Start the server
httpServer.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
}); 