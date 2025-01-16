import WebSocket from 'ws';

describe('LLMNode', () => {
    let ws;
    const WS_URL = 'ws://localhost:3000/ws';
    const LLM_NODE_ID = 'test_llm_1';
    
    beforeEach((done) => {
        ws = new WebSocket(WS_URL);
        ws.on('open', () => {
            done();
        });
    });

    afterEach((done) => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
        done();
    });

    test('should process LLM request and stream response', (done) => {
        const chunks = [];
        const prompt = "What is 1+1? Answer in one word.";

        const llmRequest = {
            type: 'llmRequest',
            nodeId: LLM_NODE_ID,
            prompt: prompt,
            model: 'smollm:latest'
        };

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            
            if (message.type === 'llmResponseChunk') {
                chunks.push(message.chunk);
            }
            
            if (message.type === 'llmResponseComplete') {
                const fullResponse = chunks.join('');
                expect(fullResponse).toBeTruthy();
                expect(fullResponse.length).toBeGreaterThan(0);
                done();
            }
        });

        ws.send(JSON.stringify(llmRequest));
    }, 10000); // 10s timeout for LLM response

    test('should handle LLM errors gracefully', (done) => {
        const invalidRequest = {
            type: 'llmRequest',
            nodeId: LLM_NODE_ID,
            prompt: "Test prompt",
            model: 'non_existent_model'
        };

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'llmError') {
                expect(message.nodeId).toBe(LLM_NODE_ID);
                expect(message.error).toBeTruthy();
                done();
            }
        });

        ws.send(JSON.stringify(invalidRequest));
    });

    test('should abort LLM request on client disconnect', (done) => {
        const longPrompt = "Write a very long story about AI. Make it at least 1000 words.";
        let responseStarted = false;

        const llmRequest = {
            type: 'llmRequest',
            nodeId: LLM_NODE_ID,
            prompt: longPrompt,
            model: 'smollm:latest'
        };

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'llmResponseChunk') {
                responseStarted = true;
                // Once we get the first chunk, close the connection
                ws.close();
            }
            if (message.type === 'llmAborted') {
                expect(responseStarted).toBe(true);
                done();
            }
        });

        ws.send(JSON.stringify(llmRequest));
    }, 15000); // 15s timeout

    test('should emit nodeOutput events', (done) => {
        const prompt = "Generate a short test output.";
        let outputReceived = false;

        const llmRequest = {
            type: 'llmRequest',
            nodeId: LLM_NODE_ID,
            prompt: prompt,
            model: 'smollm:latest'
        };

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            
            if (message.type === 'nodeOutput' && message.nodeId === LLM_NODE_ID) {
                outputReceived = true;
                expect(message.content).toBeTruthy();
                expect(typeof message.content).toBe('string');
            }
            
            if (message.type === 'llmResponseComplete' && outputReceived) {
                done();
            }
        });

        ws.send(JSON.stringify(llmRequest));
    }, 10000); // 10s timeout
}); 