import { WebSocketServer } from 'ws'; // Correct named import
import { setupBlessedUI, updateUIOnConnection, updateUIOnDisconnection, updateUILog } from './blessedUI.js';
import { handleWebSocketConnection } from './websocketHandler.js';

// Initialize the Blessed UI
const { screen, logBox, statusBox, clientsBox } = setupBlessedUI();

// Create a WebSocket server
const wss = new WebSocketServer({ port: 8080 }); // Instantiate WebSocket server correctly

// Handle WebSocket connections
wss.on('connection', (ws) => handleWebSocketConnection(ws, logBox, clientsBox, updateUIOnConnection, updateUIOnDisconnection, updateUILog));

// Start the server
const port = 3000;
const statusMessage = `Server is running on port ${port}`;
statusBox.setContent(statusMessage);
updateUILog(logBox, `[INFO] ${statusMessage}`);
screen.render();

// Allow exiting the UI with 'q', 'Ctrl+C', or 'Esc'
screen.key(['q', 'C-c', 'Esc'], () => process.exit(0));
