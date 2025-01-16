import { generateStream } from '../ollama/ollama_client.js';
import { WebSocketServer, WebSocket } from 'ws';
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const wss = new WebSocketServer({ noServer: true });
const PORT = 8080;

const server = http.createServer((req, res) => {
  if (req.url === '/') {
    const filePath = path.join(__dirname, 'index_multi.html');
    fs.readFile(filePath, 'utf8', (err, content) => {
      if (err) {
        res.writeHead(500);
        res.end(`Server Error: ${err.code}`);
      } else {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(content);
      }
    });
  } else {
    res.writeHead(404);
    res.end('Not Found');
  }
});

// Store active conversations
const activeConversations = {};

// Whitelisted commands
const allowedCommands = ['ls', 'pwd', 'date', 'whoami', 'uptime'];

async function startConversation(conversationId, ws) {
  const modelName = 'smollm';
  const toolsPrompt = `You are a linux command line tool. Output the command you want to run in JSON format: {"tool": "linux_command", "command": "command_name"} and then stop.

Available commands: ${allowedCommands.join(', ')}

Here are some examples:

User: What's the current date?
Assistant: {"tool": "linux_command", "command": "date"}

User: I want to know the current working directory.
Assistant: {"tool": "linux_command", "command": "pwd"}

User: Show me the files in the current directory.
Assistant: {"tool": "linux_command", "command": "ls"}
`;
  const userPrompt = `Im in a home/ directory. How many files are in this directory?`;

  const streamHandler = await generateStream({
    model: modelName,
    prompt: userPrompt,
    options: {
      system: toolsPrompt,
      template: `<|start_header_id|>system<|end_header_id|>{{ .System }}<|eot_id|>
<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>{"tool": "`,
    },
    modelOptions: { num_thread: 3 ,
        stop: ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '/INST'],
    }
  });

  activeConversations[conversationId] = {
    status: 'starting',
    fullMessage: "",
    streamHandler
  };

  // Notify all clients about the new conversation
  broadcast(JSON.stringify({ conversationId, status: 'starting' }));

  streamHandler.on('data', async (data) => {
    console.log(`Received data from stream:`, data);
    activeConversations[conversationId].fullMessage += data.data.response;

    // Check if the response contains a tool call
    let toolCall = null;
    try {
      toolCall = JSON.parse(data.data.response);
    } catch (error) {
      // Ignore parsing errors, it's probably not a tool call
    }

    if (toolCall && toolCall.tool === 'linux_command') {
      // Validate the command
      if (allowedCommands.includes(toolCall.command)) {
        // Execute the command
        exec(toolCall.command, (error, stdout, stderr) => {
          const commandResult = error ? `Error: ${error.message}` : stdout;
          const messageToSend = JSON.stringify({
            conversationId,
            status: 'tool_result',
            command: toolCall.command,
            result: commandResult
          });
          broadcast(messageToSend); // Stringify the message
          console.log('Sending tool result via WebSocket:', messageToSend);
        });
      } else {
        const errorMessage = JSON.stringify({
          conversationId,
          status: 'tool_error',
          command: toolCall.command,
          error: 'Command not allowed'
        });
        broadcast(errorMessage); // Stringify the message
        console.log('Sending tool error via WebSocket:', errorMessage);
      }
    } else {
      // Regular response
      const messageToSend = JSON.stringify({
        conversationId,
        status: 'generating',
        token: data.data.response,
        fullMessage: activeConversations[conversationId].fullMessage
      });
      broadcast(messageToSend); // Stringify the message
      console.log('Sending token via WebSocket:', messageToSend);
    }
  });

  streamHandler.on('end', () => {
    console.log(`Stream ended for conversation ${conversationId}.`);
    const completeMessage = JSON.stringify({ conversationId, status: 'complete' });
    // Notify all clients about completion
    broadcast(completeMessage); // Stringify the message
    console.log('Sending complete message:', completeMessage);
    delete activeConversations[conversationId];
  });

  streamHandler.on('error', (error) => {
    console.error(`Stream error for conversation ${conversationId}:`, error);
    const errorMessage = JSON.stringify({ conversationId, status: 'error', error: error.message });
    // Notify all clients about the error
    broadcast(errorMessage); // Stringify the message
    console.log('Sending error message:', errorMessage);
    delete activeConversations[conversationId];
  });
}

// Helper function to broadcast messages to all connected clients
function broadcast(message) {
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message); // Send the stringified message
    }
  });
}

server.on('upgrade', (request, socket, head) => {
  wss.handleUpgrade(request, socket, head, ws => {
    wss.emit('connection', ws, request);

    // Send initial status of all active conversations to the new client
    for (const conversationId in activeConversations) {
      ws.send(JSON.stringify({
        conversationId,
        status: activeConversations[conversationId].status,
        fullMessage: activeConversations[conversationId].fullMessage
      }));
    }
  });
});

server.listen(PORT, async () => {
  console.log(`Server started on http://localhost:${PORT}`);

  // Start conversations when the server starts
  for (let i = 1; i <= 2; i++) {
    await startConversation(i);
  }
});