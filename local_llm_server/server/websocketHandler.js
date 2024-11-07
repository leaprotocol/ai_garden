import WebSocket from 'ws';
import { Ollama } from 'ollama'
import { AbortController } from 'node-abort-controller';

//const ollama = new Ollama({ host: 'http://171.248.46.71:28108' })
const ollama = new Ollama({ host: 'http://localhost:11434' })

export function handleWebSocketConnection(ws, logBox, clientsBox, updateUIOnConnection, updateUIOnDisconnection, updateUILog) {
    let isClientConnected = true;
    let sessionId;

    ws.on('message', async (message) => {
        try {
            const data = JSON.parse(message);
            const text = data.text;
            const model = data.model;
            const options = data.options;
            sessionId = data.sessionId;


            if (!text || !sessionId) {
                ws.send(JSON.stringify({ error: 'Text and valid session ID are required' }));
                updateUILog(logBox, `[ERROR] Client with session ID ${sessionId}: Text and valid session ID are required`);
                return;
            }

            updateUIOnConnection(clientsBox, sessionId);
            updateUILog(logBox, `[INFO] Session ${sessionId}: Received query`);


            ws.on('close', () => {
                isClientConnected = false;
                updateUIOnDisconnection(clientsBox, sessionId);
                updateUILog(logBox, `[INFO] Session ${sessionId}: Client disconnected, aborting request.`);
            });

            await processQuery(ws, text, sessionId, logBox, updateUILog, model, options);

        } catch (error) {
            if (isClientConnected) {
                ws.send(JSON.stringify({ error: 'Failed to process query' }));
                updateUILog(logBox, `[ERROR] Session ${sessionId}: Failed to process query - ${error.message}`);
            }
        }
    });
}

async function processQuery(ws, text, sessionId, logBox, updateUILog, model, options) {
    try {
        // Set up the message to send to the model
        const message = { role: 'system', content: text };

        // Send the message to the model and stream the response
        const response = await ollama.generate({
            model: model,
            prompt: text,
            raw: true,
            stream: true,       // Enable streaming
            options
        });

        let accumulatedText = '';  // Accumulate the response chunks
        for await (const part of response) {
            console.log(part.response)
            if (ws.readyState !== WebSocket.OPEN) {
                response.abort()
            }

            try {
                updateUILog(logBox, JSON.stringify(part.response));
                accumulatedText += part.response;  // Accumulate chunks
                ws.send(JSON.stringify({ sessionId, textChunk: part.response }));
                updateUILog(logBox, `[INFO] Session ${sessionId}: Sent part of the result to client`);
            } catch (parseError) {
                ws.send(JSON.stringify({ error: 'Internal server error' }));
                updateUILog(logBox, `[ERROR] Session ${sessionId}: Internal server error during processing ${parseError}`);
            }
        }

        // Optionally, send the final accumulated text if needed
        updateUILog(logBox, `[INFO] Session ${sessionId} completed`);

        try {
            ws.send(JSON.stringify({ sessionId, textChunk: null, type: "completed" }));
            updateUILog(logBox, `[INFO] Session ${sessionId}: Sent completed status to client`);
        } catch (parseError) {
            ws.send(JSON.stringify({ error: 'Internal server error' }));
            updateUILog(logBox, `[ERROR] Session ${sessionId}: Internal server error during processing`);
        }

        return accumulatedText;

    } catch (error) {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ error: 'Stream error during processing' }));
            updateUILog(logBox, `[ERROR] Session ${sessionId}: Stream error during processing - ${error} ${error.stack}`);
        }
    }
}
