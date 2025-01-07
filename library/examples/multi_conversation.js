import { generateStream } from '../ollama/ollama_client.js';
import { WebSocketServer } from 'ws';
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

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

wss.on('connection', ws => {
  console.log('Client connected');

  async function startConversation(ws, conversationId) {
    const modelName = 'smollm:1.7b-instruct-v0.2-q6_K';
    const systemPrompt = `You are a helpful assistant. Respond to the prompt in a single sentence.`;
    const userPrompt = `Conversation ${conversationId}`;

    const streamHandler = await generateStream({
      model: modelName,
      prompt: userPrompt,
      options: {
        system: systemPrompt,
        template: "{{ .System }}<|eot_id|>{{ .Prompt }}"
      },
      modelOptions: { num_thread: 3 }
    });

    ws.send(JSON.stringify({ conversationId, status: 'starting' }));

    let fullMessage = "";

    streamHandler.on('data', (data) => {
      console.log(`Received data from stream:`, data);
      fullMessage += data.data.response;
      const messageToSend = JSON.stringify({
        conversationId,
        status: 'generating',
        token: data.data.response,
        fullMessage: fullMessage
      });
      ws.send(messageToSend);
      console.log('Sending token via WebSocket:', messageToSend);
    });

    streamHandler.on('end', () => {
      console.log(`Stream ended for conversation ${conversationId}.`);
      const completeMessage = JSON.stringify({ conversationId, status: 'complete' });
      ws.send(completeMessage);
      console.log('Sending complete message:', completeMessage);
    });

    streamHandler.on('error', (error) => {
      console.error(`Stream error for conversation ${conversationId}:`, error);
      const errorMessage = JSON.stringify({ conversationId, status: 'error', error: error.message });
      ws.send(errorMessage);
      console.log('Sending error message:', errorMessage);
    });
  }

  async function main(ws) {
    for (let i = 1; i <= 3; i++) {
      startConversation(ws, i);
    }
  }

  main(ws);

  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

server.on('upgrade', (request, socket, head) => {
  wss.handleUpgrade(request, socket, head, socket => {
    wss.emit('connection', socket, request);
  });
});

server.listen(PORT, () => {
  console.log(`Server started on http://localhost:${PORT}`);
}); 