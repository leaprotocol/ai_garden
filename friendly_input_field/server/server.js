import express from 'express';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createOllamaClient } from './ollama.js';

const LOG_LEVELS = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3
};

const CURRENT_LOG_LEVEL = LOG_LEVELS.DEBUG;

function log(level, ...args) {
    if (level >= CURRENT_LOG_LEVEL) {
        const prefix = Object.keys(LOG_LEVELS).find(key => LOG_LEVELS[key] === level);
        console.log(`[${prefix}]`, ...args);
    }
}

// Set up Express with static file serving
const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const httpServer = createServer(app);

// Serve static files from the client directory
app.use(express.static(join(__dirname, '../client'), {
    setHeaders: (res, path) => {
        if (path.endsWith('.js')) {
            res.setHeader('Content-Type', 'application/javascript');
        } else if (path.endsWith('.css')) {
            res.setHeader('Content-Type', 'text/css');
        }
    }
}));

// Create WebSocket server attached to HTTP server
const wss = new WebSocketServer({ server: httpServer });
const ollama = createOllamaClient();

// Track active LLM processes
const ollamaProcesses = new Map();

async function getAvailableModels() {
    try {
        const models = await ollama.list();
        return models.models;
    } catch (error) {
        log(LOG_LEVELS.ERROR, 'Failed to fetch models:', error);
        return [];
    }
}

async function handleLLMRequest(ws, events, model) {
    const requestId = crypto.randomUUID();
    const timing = {
        clientTimestamp: events.clientTimestamp,
        serverReceived: performance.now(),
        ollamaStart: null,
        ollamaFirstChunk: null
    };
    
    try {
        // Send acknowledgment of request receipt
        ws.send(JSON.stringify({
            type: 'timing_update',
            requestId,
            event: 'server_received',
            timestamp: timing.serverReceived
        }));

        const prompt = `You are analyzing user input events...`;

        // Record Ollama start time
        timing.ollamaStart = performance.now();
        ws.send(JSON.stringify({
            type: 'timing_update',
            requestId,
            event: 'ollama_start',
            timestamp: timing.ollamaStart
        }));

        const response = await ollama.generate({
            model: model,
            prompt: prompt,
            stream: true
        });

        let hasReceivedFirstChunk = false;
        for await (const chunk of response) {
            if (chunk.response) {
                if (!hasReceivedFirstChunk) {
                    timing.ollamaFirstChunk = performance.now();
                    hasReceivedFirstChunk = true;
                    
                    ws.send(JSON.stringify({
                        type: 'timing_update',
                        requestId,
                        event: 'ollama_first_chunk',
                        timestamp: timing.ollamaFirstChunk
                    }));
                }

                ws.send(JSON.stringify({
                    type: 'llm_response',
                    text: chunk.response,
                    requestId
                }));
            }
        }

        // Send completion with timing data
        ws.send(JSON.stringify({
            type: 'llm_complete',
            requestId,
            timing: {
                serverProcessing: timing.ollamaStart - timing.serverReceived,
                ollamaProcessing: timing.ollamaFirstChunk - timing.ollamaStart,
                totalServerTime: performance.now() - timing.serverReceived
            }
        }));

    } catch (error) {
        log(LOG_LEVELS.ERROR, `Error in request ${requestId}:`, error);
        ws.send(JSON.stringify({
            type: 'error',
            text: error.message,
            requestId
        }));
    }
}

wss.on('connection', async (ws) => {
    log(LOG_LEVELS.INFO, 'Client connected');

    // Send available models to client
    try {
        const models = await getAvailableModels();
        ws.send(JSON.stringify({
            type: 'available_models',
            models: models
        }));
    } catch (error) {
        log(LOG_LEVELS.ERROR, 'Error sending models to client:', error);
    }

    ws.on('message', async data => {
        try {
            const rawMessage = data;
            log(LOG_LEVELS.DEBUG, 'Raw WebSocket message:', rawMessage);

            // Parse the message only once
            const message = JSON.parse(rawMessage);
            const messageType = message.type;

            log(LOG_LEVELS.INFO, 'Parsed message type:', messageType);
            log(LOG_LEVELS.INFO, 'Full message stringified:', JSON.stringify(message));

            switch (messageType) {
                case 'event_batch':
                    log(LOG_LEVELS.INFO, 'Processing event batch with model:', message.model);
                    if (!message.model) {
                        log(LOG_LEVELS.ERROR, 'No model specified in event batch');
                        ws.send(JSON.stringify({
                            type: 'error',
                            text: 'No model specified'
                        }));
                        return;
                    }
                    // Ensure events are parsed correctly
                    const events = message.events;
                    await handleLLMRequest(ws, events, message.model);
                    break;

                case 'getModels':
                    const models = await getAvailableModels();
                    ws.send(JSON.stringify({
                        type: 'available_models',
                        models: models
                    }));
                    break;

                default:
                    log(LOG_LEVELS.WARN, `Unhandled message type: ${messageType}`);
                    break;
            }
        } catch (error) {
            log(LOG_LEVELS.ERROR, 'WebSocket message error:', error);
            ws.send(JSON.stringify({
                type: 'error',
                text: 'Invalid message format',
                details: error.message
            }));
        }
    });

    ws.on('error', error => {
        log(LOG_LEVELS.ERROR, 'WebSocket error:', error);
    });
});

// Start the server
const PORT = 3000;
httpServer.listen(PORT, () => {
    log(LOG_LEVELS.INFO, `Server running on http://localhost:${PORT}`);
});

// Handle graceful shutdown
process.on('SIGTERM', () => gracefulShutdown());
process.on('SIGINT', () => gracefulShutdown());

function gracefulShutdown() {
    log(LOG_LEVELS.INFO, '\nShutting down server...');
    if (wss.clients) {
        wss.clients.forEach(client => client.close());
    }
    httpServer.close(() => {
        log(LOG_LEVELS.INFO, 'Server closed');
        process.exit(0);
    });
}

app.get('/api/models', async (req, res) => {
    try {
        const models = await getAvailableModels();
        res.json(models);
    } catch (error) {
        res.status(500).send('Error fetching models');
    }
}); 