import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { WebSocketServer } from 'ws';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();

// Serve static files from the root of llm-graph2
app.use(express.static(join(__dirname, '../')));

const server = app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});

// Initialize WebSocket Server
const wss = new WebSocketServer({ server });

// Handle WebSocket connections
wss.on('connection', (ws) => {
    console.log('New WebSocket connection');

    ws.on('message', (message) => {
        console.log('Received:', message);
        // Handle incoming messages and route to appropriate nodes
        // Example: Broadcast to all connected clients
        wss.clients.forEach(client => {
            if (client !== ws && client.readyState === ws.OPEN) {
                client.send(message);
            }
        });
    });

    ws.on('close', () => {
        console.log('WebSocket connection closed');
    });
});

// Handle graceful shutdown
process.on('SIGTERM', () => gracefulShutdown(wss, server));
process.on('SIGINT', () => gracefulShutdown(wss, server));

function gracefulShutdown(wss, server) {
    console.log('\nShutting down server...');
    if (wss && wss.clients) {
        wss.clients.forEach(client => {
            client.close();
        });
    }
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
} 