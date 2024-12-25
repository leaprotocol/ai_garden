import WebSocket from 'ws';

// Configuration
const WS_URL = 'ws://localhost:3000/ws';
const SPAWN_INTERVAL = 5000; // 5 seconds
const MAX_SPAWNS = 3;
const CLEANUP_TIMEOUT = 30000; // 30 seconds

// Test node IDs
const SPAWNER_ID = 'spawner_test_1';
const BUCKET_ID = 'bucket_test_1';

// Prompts to test different aspects of event flow
const PROMPTS = [
    {
        role: "Event Generator 1",
        prompt: "Generate a short event about AI progress. Keep it to 1-2 sentences."
    },
    {
        role: "Event Generator 2",
        prompt: "Generate a short event about AI ethics. Keep it to 1-2 sentences."
    },
    {
        role: "Event Generator 3",
        prompt: "Generate a short event about AI impact. Keep it to 1-2 sentences."
    }
];

let currentPromptIndex = 0;
let activeSpawns = 0;

// Create WebSocket connection
const ws = new WebSocket(WS_URL);

// Track node states
let spawnedNodes = new Map(); // Track spawned nodes and their states
let bucketEvents = []; // Track events received by bucket

function log(level, ...args) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${level}]`, ...args);
}

// Handle WebSocket events
ws.on('open', () => {
    log('INFO', 'Connected to WebSocket server');
    log('INFO', 'Starting spawner -> bucket flow test...\n');

    // Create leaky bucket node
    createBucketNode();

    // Start periodic spawning after bucket is created
    setTimeout(() => {
        const spawnInterval = setInterval(() => {
            if (activeSpawns < MAX_SPAWNS) {
                requestSpawn();
            }
        }, SPAWN_INTERVAL);

        // Cleanup after 2 minutes
        setTimeout(() => {
            clearInterval(spawnInterval);
            log('INFO', '\nTest complete. Closing connection...');
            // Log final bucket state
            log('INFO', `Final bucket events count: ${bucketEvents.length}`);
            log('DEBUG', 'Bucket events:', bucketEvents);
            ws.close();
        }, 120000); // 2 minutes
    }, 1000); // Wait 1s for bucket creation
});

ws.on('message', (data) => {
    const message = JSON.parse(data);
    handleMessage(message);
});

ws.on('close', () => {
    log('INFO', 'WebSocket connection closed');
    process.exit(0);
});

// Create the leaky bucket node
function createBucketNode() {
    const bucketNode = {
        type: 'createNode',
        nodeType: 'leakyBucket',
        nodeId: BUCKET_ID,
        position: { x: 600, y: 200 },
        data: {
            label: 'Test Bucket',
            bucketSize: 5,
            leakInterval: 10000
        }
    };
    ws.send(JSON.stringify(bucketNode));
    log('INFO', 'Created leaky bucket node');
}

// Handle different message types
function handleMessage(message) {
    switch (message.type) {
        case 'nodeSpawned':
            if (message.parentNodeId === SPAWNER_ID) {
                log('INFO', `[SPAWN] New node created: ${message.spawnedNodeId}`);
                spawnedNodes.set(message.spawnedNodeId, { status: 'spawned' });
                activeSpawns++;
                
                // Send LLM request to the spawned node
                const promptData = PROMPTS[currentPromptIndex % PROMPTS.length];
                sendLLMRequest(message.spawnedNodeId, promptData);
                currentPromptIndex++;
            }
            break;

        case 'nodeCompleted':
            if (spawnedNodes.has(message.nodeId)) {
                log('INFO', `[COMPLETE] Node finished: ${message.nodeId}`);
                spawnedNodes.delete(message.nodeId);
                activeSpawns--;
            }
            break;

        case 'llmResponseChunk':
            if (spawnedNodes.has(message.nodeId)) {
                process.stdout.write('.');
                const node = spawnedNodes.get(message.nodeId);
                node.response = (node.response || '') + message.chunk;
                spawnedNodes.set(message.nodeId, node);
            }
            break;

        case 'llmResponseComplete':
            if (spawnedNodes.has(message.nodeId)) {
                const node = spawnedNodes.get(message.nodeId);
                log('INFO', `\n[RESPONSE] Node ${message.nodeId}:`);
                log('DEBUG', node.response);
                log('INFO', '-------------------');
            }
            break;

        case 'bucketEvent':
            if (message.nodeId === BUCKET_ID) {
                bucketEvents.push({
                    timestamp: message.timestamp,
                    content: message.content
                });
                log('INFO', `[BUCKET] Received event #${bucketEvents.length}`);
                log('DEBUG', `Event content: ${message.content}`);
            }
            break;

        case 'llmError':
            if (spawnedNodes.has(message.nodeId)) {
                log('ERROR', `\n[ERROR] Node ${message.nodeId}:`, message.error);
            }
            break;
    }
}

// Request a new node spawn
function requestSpawn() {
    const spawnRequest = {
        type: 'spawnNode',
        nodeId: SPAWNER_ID,
        nodeType: 'llm',
        timeout: CLEANUP_TIMEOUT,
        timestamp: Date.now(),
        formattedTime: new Date().toLocaleTimeString(),
        promptData: PROMPTS[currentPromptIndex % PROMPTS.length]
    };
    ws.send(JSON.stringify(spawnRequest));
}

// Send LLM request to a spawned node
function sendLLMRequest(nodeId, promptData) {
    const llmRequest = {
        type: 'llmRequest',
        nodeId: nodeId,
        prompt: promptData.prompt,
        model: 'smollm:latest'
    };
    ws.send(JSON.stringify(llmRequest));
    log('INFO', `[PROMPT] Node ${nodeId} (${promptData.role}):`, promptData.prompt);
} 