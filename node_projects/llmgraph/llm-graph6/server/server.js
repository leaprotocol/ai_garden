import express from 'express';
import { WebSocketServer } from 'ws';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import crypto from 'crypto';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = 3000;

// Enable CORS for development
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

app.use(express.static(join(__dirname, '..')));

const server = app.listen(PORT, () => {
  console.log(`HTTP server listening on http://localhost:${PORT}`);
});

const wss = new WebSocketServer({ server });

const nodes = new Map();

function broadcast(message) {
  wss.clients.forEach(client => {
    try {
      client.send(JSON.stringify(message));
    } catch (error) {
      console.error('Error sending message to client:', error);
    }
  });
}

function broadcastNodes() {
  broadcast({
    type: 'nodesUpdated',
    nodes: Array.from(nodes.entries()),
    connections: []
  });
}

// Handle graceful shutdown
function cleanup() {
  console.log('Cleaning up...');
  wss.clients.forEach(client => {
    client.close();
  });
  server.close(() => {
    console.log('Server shut down');
    process.exit(0);
  });
}

process.on('SIGTERM', cleanup);
process.on('SIGINT', cleanup);

wss.on('connection', ws => {
  console.log('Client connected');
  
  try {
    broadcastNodes();
  } catch (error) {
    console.error('Error during initial broadcast:', error);
  }

  ws.on('message', message => {
    try {
      const event = JSON.parse(message.toString());
      console.log('Received message:', event);

      switch (event.type) {
        case 'createNode':
          const nodeId = crypto.randomUUID();
          nodes.set(nodeId, { id: nodeId, type: event.nodeType, position: event.position });
          broadcastNodes();
          break;
        default:
          console.log('Unknown message type:', event.type);
      }
    } catch (error) {
      console.error('Error handling message:', error);
      ws.send(JSON.stringify({ 
        type: 'error', 
        message: 'Failed to process your request.' 
      }));
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });

  ws.on('error', error => {
    console.error('WebSocket error:', error);
  });
}); 