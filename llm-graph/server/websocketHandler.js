import { WebSocket, WebSocketServer } from 'ws';
import { Ollama } from 'ollama';
import { exec } from 'child_process';
import { processSource } from './sourceProcessor.js';

const ollama = new Ollama({
  host: 'http://localhost:11434'
});

export function setupWebSocket(server) {
  const wss = new WebSocketServer({ server });

  wss.on('connection', async (ws) => {
    try {
      const models = await ollama.list();
      ws.send(JSON.stringify({
        type: 'models',
        models: models.models
      }));
    } catch (error) {
      console.error('Error fetching models:', error);
      ws.send(JSON.stringify({
        type: 'error',
        message: 'Failed to fetch models'
      }));
    }
    
    ws.on('message', async (message) => {
      try {
        const data = JSON.parse(message);
        
        switch (data.type) {
          case 'getModels':
            const models = await ollama.list();
            ws.send(JSON.stringify({
              type: 'models',
              models: models.models
            }));
            break;

          case 'processText':
            const response = await ollama.generate({
              model: data.model,
              prompt: data.prompt + (data.text || data.chunk || ''),
              stream: true
            });

            for await (const chunk of response) {
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                  type: 'nodeStream',
                  nodeId: data.nodeId,
                  chunk: chunk.response
                }));
              }
            }

            ws.send(JSON.stringify({
              type: 'nodeComplete',
              nodeId: data.nodeId
            }));
            break;

          case 'processSource':
            try {
              const chunks = await processSource(data.path, {
                sourceType: data.sourceType,
                ...data.options,
                onProgress: (progress) => {
                  ws.send(JSON.stringify({
                    type: 'progress',
                    nodeId: data.nodeId,
                    progress
                  }));
                }
              });

              for (const chunk of chunks) {
                if (ws.readyState === WebSocket.OPEN) {
                  ws.send(JSON.stringify({
                    type: 'nodeStream',
                    nodeId: data.nodeId,
                    chunk
                  }));
                }
              }

              ws.send(JSON.stringify({
                type: 'nodeComplete',
                nodeId: data.nodeId
              }));
            } catch (error) {
              ws.send(JSON.stringify({
                type: 'error',
                nodeId: data.nodeId,
                message: error.message
              }));
            }
            break;

          case 'executeCommand':
            exec(data.command, (error, stdout, stderr) => {
              if (error) {
                ws.send(JSON.stringify({
                  type: 'error',
                  nodeId: data.nodeId,
                  message: error.message
                }));
                return;
              }

              ws.send(JSON.stringify({
                type: 'nodeStream',
                nodeId: data.nodeId,
                chunk: stdout
              }));

              if (stderr) {
                ws.send(JSON.stringify({
                  type: 'error',
                  nodeId: data.nodeId,
                  message: stderr
                }));
              }

              ws.send(JSON.stringify({
                type: 'nodeComplete',
                nodeId: data.nodeId
              }));
            });
            break;
        }
      } catch (error) {
        console.error('WebSocket error:', error);
        ws.send(JSON.stringify({
          type: 'error',
          message: error.message
        }));
      }
    });
  });

  return wss;
} 