import WebSocket from 'ws';
import { v4 as uuidv4 } from 'uuid';
import {screen, createThreadBox, logToConsole} from './ui.js';

// Function to create a WebSocket connection for each thread
export function createThread(threadId, userPrompt, options) {
    const sessionId = uuidv4();
    const ws = new WebSocket('ws://localhost:8080');
    let resolvePromise, rejectPromise;
    const result = new Promise((resolve, reject) => {resolvePromise = resolve;rejectPromise = reject;});

    // Create the full prompt using the provided template
    const fullPrompt = userPrompt;

    // Create a UI box for this thread's output
    const threadBox = createThreadBox(threadId);
    threadBox.on('click', function() {
        threadBox.focus();
    });

    let outputText = '';  // Initialize an empty string to accumulate the response

    ws.on('open', () => {
        threadBox.setContent(`Connected to server with session ID: ${sessionId}\n\nSending request...\n\n${userPrompt}`);
        screen.render();

        // Send a message to the server
        ws.send(JSON.stringify({ sessionId, text: fullPrompt, model:options.model, num_thread:options.num_thread }));
    });

    ws.on('message', (message) => {
        const data = JSON.parse(message);
        if (data.error) {
            threadBox.setContent(`${threadBox.getContent()}\n\nError: ${data.error}`);
            rejectPromise(data.error);
        } else if (data.type==="completed") {
            threadBox.style.border.fg = "green"
            screen.render();
            resolvePromise({result: outputText, sessionId, fullPrompt});
        }
        else if (data.textChunk) {
            outputText += data.textChunk;
            //outputText = outputText/*.replace(/\s+/g, ' ')*/.trim();
            threadBox.setContent(`${threadBox.getContent().split('Received result:')[0]}Received result:\n${outputText}`);
        }
        screen.render();
    });

    ws.on('close', () => {
        rejectPromise('close')
        threadBox.setContent(`${threadBox.getContent()}\n\nDisconnected from server.`);
        screen.render();
    });

    ws.on('error', (error) => {
        rejectPromise(error)
        threadBox.setContent(`${threadBox.getContent()}\n\nWebSocket error: ${error}`);
        screen.render();
    });
    return result;
}
