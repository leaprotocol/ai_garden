import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { WebSocketServer } from 'ws';
import { setupWebSocket } from './websocketHandler.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();

// Serve static files with proper MIME types
app.use(express.static(join(__dirname, '../'), {
    setHeaders: (res, path) => {
        if (path.endsWith('.js')) {
            res.setHeader('Content-Type', 'application/javascript');
        } else if (path.endsWith('.css')) {
            res.setHeader('Content-Type', 'text/css');
        }
    }
}));

const server = app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});

const wss = setupWebSocket(server);

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