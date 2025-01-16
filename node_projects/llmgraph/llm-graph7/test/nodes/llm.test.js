import { expect } from 'chai';
import WebSocket from 'ws';

const WS_URL = 'ws://localhost:3000/ws';
const LLM_NODE_ID = 'test_llm_1';

describe('LLM Node', () => {
    let ws;
    let receivedChunks = [];
    let isComplete = false;

    before((done) => {
        ws = new WebSocket(WS_URL);
        ws.on('open', () => {
            console.log('Connected to WebSocket server');
            createLLMNode();
            done();
        });

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.nodeId === LLM_NODE_ID) {
                switch (message.type) {
                    case 'llmResponseChunk':
                        receivedChunks.push(message.chunk);
                        break;
                    case 'llmResponseComplete':
                        isComplete = true;
                        break;
                }
            }
        });
    });

    after(() => {
        ws.close();
    });

    beforeEach(() => {
        receivedChunks = [];
        isComplete = false;
    });

    function createLLMNode() {
        const llmNode = {
            type: 'createNode',
            nodeType: 'llm',
            nodeId: LLM_NODE_ID,
            data: {
                label: 'Test LLM',
                model: 'smollm:latest'
            }
        };
        ws.send(JSON.stringify(llmNode));
    }

    function sendPrompt(prompt) {
        const request = {
            type: 'llmRequest',
            nodeId: LLM_NODE_ID,
            prompt,
            model: 'smollm:latest'
        };
        ws.send(JSON.stringify(request));
    }

    it('should create an LLM node', (done) => {
        setTimeout(() => {
            expect(ws.readyState).to.equal(WebSocket.OPEN);
            done();
        }, 1000);
    });

    it('should process a simple prompt', (done) => {
        sendPrompt('Say hello in one word.');
        
        setTimeout(() => {
            expect(receivedChunks.length).to.be.greaterThan(0);
            expect(isComplete).to.be.true;
            const response = receivedChunks.join('');
            expect(response.toLowerCase()).to.include('hello');
            done();
        }, 5000);
    });

    it('should handle streaming responses', (done) => {
        const timestamps = [];
        const originalPush = receivedChunks.push.bind(receivedChunks);
        receivedChunks.push = (...args) => {
            timestamps.push(Date.now());
            return originalPush(...args);
        };

        sendPrompt('Count from 1 to 5 slowly.');

        setTimeout(() => {
            expect(timestamps.length).to.be.greaterThan(1);
            // Check if chunks arrived at different times
            for (let i = 1; i < timestamps.length; i++) {
                expect(timestamps[i] - timestamps[i-1]).to.be.greaterThan(0);
            }
            done();
        }, 10000);
    });

    it('should handle errors gracefully', (done) => {
        let errorReceived = false;
        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'llmError' && message.nodeId === LLM_NODE_ID) {
                errorReceived = true;
            }
        });

        // Send an invalid request to trigger an error
        ws.send(JSON.stringify({
            type: 'llmRequest',
            nodeId: LLM_NODE_ID,
            // Missing required prompt field
            model: 'invalid_model'
        }));

        setTimeout(() => {
            expect(errorReceived).to.be.true;
            done();
        }, 2000);
    });
}); 