import express from 'express';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import crypto from 'crypto';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();

// Serve static files from current directory
app.use(express.static(__dirname));

const server = app.listen(3000, () => {
  console.log('HTTP Server running on http://localhost:3000');
});

// Initialize WebSocket Server attached to HTTP server
const wss = new WebSocketServer({ server });

const nodes = new Map();
const connections = new Map();
const nodeIntervals = new Map();

function broadcast(message) {
  wss.clients.forEach(client => {
    client.send(JSON.stringify(message));
  });
}

function broadcastNodes() {
  broadcast({
    type: 'nodesUpdated',
    nodes: Object.fromEntries(nodes),
    connections: Object.fromEntries(connections)
  });
}

wss.on('connection', ws => {
  console.log('Client connected');

  ws.on('message', message => {
    const event = JSON.parse(message.toString());
    console.log('Received event:', event);

    switch (event.type) {
      case 'getNodes':
        broadcastNodes();
        break;

      case 'createNode':
        const nodeId = crypto.randomUUID().slice(0, 8);
        nodes.set(nodeId, {
          type: event.nodeType,
          position: event.position,
          data: {}
        });
        broadcastNodes();
        break;

      case 'toggleTimer':
        if (nodes.has(event.nodeId)) {
          if (nodeIntervals.has(event.nodeId)) {
            clearInterval(nodeIntervals.get(event.nodeId));
            nodeIntervals.delete(event.nodeId);
          } else {
            const interval = setInterval(() => {
              broadcast({
                type: 'nodeEvent',
                target: event.nodeId,
                payload: { timestamp: Date.now() }
              });
            }, 1000);
            nodeIntervals.set(event.nodeId, interval);
          }
        }
        break;

      case 'updateInterval':
        if (nodes.has(event.nodeId)) {
          const node = nodes.get(event.nodeId);
          node.interval = parseInt(event.interval);
          nodes.set(event.nodeId, node);
          if (nodeIntervals.has(event.nodeId)) {
            clearInterval(nodeIntervals.get(event.nodeId));
            const newInterval = setInterval(() => {
              broadcast({
                type: 'nodeEvent',
                target: event.nodeId,
                payload: { timestamp: Date.now() }
              });
            }, node.interval);
            nodeIntervals.set(event.nodeId, newInterval);
          }
        }
        break;

      case 'clearGraph':
        nodes.clear();
        connections.clear();
        nodeIntervals.forEach(interval => clearInterval(interval));
        nodeIntervals.clear();
        broadcastNodes();
        break;
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

// Handle graceful shutdown
process.on('SIGTERM', () => gracefulShutdown());
process.on('SIGINT', () => gracefulShutdown());

function gracefulShutdown() {
  console.log('\nShutting down server...');
  nodeIntervals.forEach(interval => clearInterval(interval));
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
} 