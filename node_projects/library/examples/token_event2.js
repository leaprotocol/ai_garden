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
    const filePath = path.join(__dirname, 'index.html');
    fs.readFile(filePath, 'utf8', (err, content) => {
      if (err) {
        res.writeHead(500);
        res.end(`Server Error: ${err.code}`);
      } else {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(content);
      }
    });
  } else if (req.url === '/styles.css') {
    const filePath = path.join(__dirname, 'styles.css');
    fs.readFile(filePath, 'utf8', (err, content) => {
      if (err) {
        res.writeHead(404);
        res.end(`File Not Found`);
      } else {
        res.writeHead(200, { 'Content-Type': 'text/css' });
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

  async function main(ws) {
    const modelName = 'smollm';
    const systemPrompt = "You are philosophers, Hegel and Marcus Aurelius, having a debate about determinism and free will. Keep responses complete and avoid repeating previous statements. Each response should be a distinct thought progressing the conversation.";

    function conversationBuilder(history, currentActor) {
      const formattedHistory = history.map(item => `${item.speaker}: ${item.message}`).join("\n\n");
      return formattedHistory + "\n\n" + currentActor + ": ";
    }

    const history = [];
    const philosophers = ['Hegel', 'Marcus Aurelius'];
    let currentPhilosopher = 0;

    async function generateNextResponse() {
      const currentSpeaker = philosophers[currentPhilosopher];
      console.log(`Generating next response for: ${currentSpeaker}`);
      const streamHandler = await generateStream({
        model: modelName,
        prompt: conversationBuilder(history, currentSpeaker),
        options: {
          system: systemPrompt,
          template: "{{ .System }}<|eot_id|>{{ .Prompt }}"
        },
        modelOptions: { num_thread: 3 }
      });

      let currentMessage = '';
      let responseComplete = false;

      streamHandler.on('data', (data) => {
        console.log(`Received data from stream:`, data);
        currentMessage += data.data.response;
        const messageToSend = JSON.stringify({ 
          speaker: currentSpeaker, 
          status: 'generating', 
          token: data.data.response
        });
        if (data.tokenCount >= 70) {
          streamHandler.abort();
          console.log('Aborting stream due to token limit.');
        }
        ws.send(messageToSend);
        console.log('Sending token via WebSocket:', messageToSend);
      });

      ws.send(JSON.stringify({ speaker: currentSpeaker, status: 'generating' }));
      console.log(`Sent 'generating' status for: ${currentSpeaker}`);


      streamHandler.on('abort', () => {
        history.push({ speaker: currentSpeaker, message: currentMessage });
        currentPhilosopher = (currentPhilosopher + 1) % 2;
        responseComplete = true;
        const completeMessage = JSON.stringify({ speaker: currentSpeaker, status: 'complete', message: currentMessage });
        ws.send(completeMessage);
        console.log('Sending complete message (abort):', completeMessage);

        if (history.length < 6) {
          generateNextResponse();
        } else {
          ws.send(JSON.stringify({ status: 'dialogueComplete' }));
          console.log('\n\n-- Dialogue Complete --');
        }
      });

      streamHandler.on('end', () => {
        console.log('Stream ended.');
        if (!responseComplete) {
          history.push({ speaker: currentSpeaker, message: currentMessage });
          currentPhilosopher = (currentPhilosopher + 1) % 2;
          const completeMessage = JSON.stringify({ speaker: currentSpeaker, status: 'complete', message: currentMessage });
          ws.send(completeMessage);
          console.log('Sending complete message (end):', completeMessage);

          if (history.length < 6) {
            generateNextResponse();
          } else {
            ws.send(JSON.stringify({ status: 'dialogueComplete' }));
            console.log('\n\n-- Dialogue Complete --');
          }
        }
      });

      streamHandler.on('error', (error) => {
        console.error('Stream error:', error);
        const errorMessage = JSON.stringify({ speaker: currentSpeaker, status: 'error', error: error.message });
        ws.send(errorMessage);
        console.log('Sending error message:', errorMessage);
      });
    }

    // Start the dialogue
    generateNextResponse();
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