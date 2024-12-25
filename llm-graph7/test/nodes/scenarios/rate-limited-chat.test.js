import { expect } from 'chai';
import WebSocket from 'ws';

const WS_URL = 'ws://localhost:3000/ws';

describe('Rate Limited Chat Scenario', () => {
    let ws;
    let nodes = {};
    let receivedEvents = [];

    before((done) => {
        ws = new WebSocket(WS_URL);
        ws.on('open', () => {
            console.log('Connected to WebSocket server');
            setupNodes();
            done();
        });

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            receivedEvents.push({
                ...message,
                timestamp: Date.now()
            });
            
            if (message.type === 'nodeCreated') {
                nodes[message.nodeId] = message.nodeType;
            }
        });
    });

    after(() => {
        ws.close();
    });

    beforeEach(() => {
        receivedEvents = [];
    });

    function setupNodes() {
        // Create LLM node
        ws.send(JSON.stringify({
            type: 'createNode',
            nodeType: 'llm',
            nodeId: 'llm_1',
            data: {
                label: 'Chat LLM',
                model: 'smollm:latest',
                prompt: 'You are a helpful assistant. Respond to: ${input}'
            }
        }));

        // Create LeakyBucket node for rate limiting
        ws.send(JSON.stringify({
            type: 'createNode',
            nodeType: 'leakyBucket',
            nodeId: 'bucket_1',
            data: {
                label: 'Rate Limiter',
                bucketSize: 2,
                leakInterval: 5000 // 5 seconds between responses
            }
        }));

        // Create Output node
        ws.send(JSON.stringify({
            type: 'createNode',
            nodeType: 'output',
            nodeId: 'output_1',
            data: {
                label: 'Chat Output'
            }
        }));

        // Connect nodes
        setTimeout(() => {
            connectNodes();
        }, 1000);
    }

    function connectNodes() {
        // LLM -> LeakyBucket
        ws.send(JSON.stringify({
            type: 'connect',
            sourceId: 'llm_1',
            targetId: 'bucket_1'
        }));

        // LeakyBucket -> Output
        ws.send(JSON.stringify({
            type: 'connect',
            sourceId: 'bucket_1',
            targetId: 'output_1'
        }));
    }

    function sendChatMessage(message) {
        ws.send(JSON.stringify({
            type: 'llmRequest',
            nodeId: 'llm_1',
            prompt: message
        }));
    }

    it('should create all required nodes', (done) => {
        setTimeout(() => {
            expect(Object.keys(nodes).length).to.equal(3);
            expect(nodes['llm_1']).to.equal('llm');
            expect(nodes['bucket_1']).to.equal('leakyBucket');
            expect(nodes['output_1']).to.equal('output');
            done();
        }, 2000);
    });

    it('should process chat messages with rate limiting', (done) => {
        // Send multiple messages in quick succession
        sendChatMessage('Hello!');
        sendChatMessage('How are you?');
        sendChatMessage('What is your name?');
        
        setTimeout(() => {
            const bucketEvents = receivedEvents.filter(
                evt => evt.nodeId === 'bucket_1' && evt.type === 'bucketUpdate'
            );
            
            expect(bucketEvents.length).to.be.greaterThan(0);
            
            // Verify bucket never exceeds size
            bucketEvents.forEach(evt => {
                expect(evt.count).to.be.lessThanOrEqual(2);
            });
            
            done();
        }, 10000);
    });

    it('should maintain minimum time between responses', (done) => {
        const outputEvents = receivedEvents.filter(
            evt => evt.nodeId === 'output_1' && evt.type === 'output'
        );
        
        if (outputEvents.length >= 2) {
            // Check time differences between consecutive outputs
            for (let i = 1; i < outputEvents.length; i++) {
                const timeDiff = outputEvents[i].timestamp - outputEvents[i-1].timestamp;
                expect(timeDiff).to.be.at.least(4500); // Allow 500ms buffer from 5000ms target
            }
        }
        
        done();
    });

    it('should handle streaming responses', (done) => {
        sendChatMessage('Tell me a long story');
        
        setTimeout(() => {
            const streamEvents = receivedEvents.filter(
                evt => evt.nodeId === 'llm_1' && evt.type === 'llmResponseChunk'
            );
            
            expect(streamEvents.length).to.be.greaterThan(1);
            
            // Verify streaming chunks
            let totalContent = '';
            streamEvents.forEach(evt => {
                expect(evt.chunk).to.be.a('string');
                totalContent += evt.chunk;
            });
            
            expect(totalContent.length).to.be.greaterThan(0);
            
            done();
        }, 15000);
    });

    it('should handle errors gracefully', (done) => {
        // Send an invalid request
        ws.send(JSON.stringify({
            type: 'llmRequest',
            nodeId: 'llm_1',
            // Missing prompt field
        }));
        
        setTimeout(() => {
            const errorEvents = receivedEvents.filter(
                evt => evt.type === 'error'
            );
            
            expect(errorEvents.length).to.be.greaterThan(0);
            expect(errorEvents[0].error).to.be.a('string');
            
            done();
        }, 2000);
    });

    it('should maintain consistent output order', (done) => {
        // Clear previous events
        receivedEvents = [];
        
        // Send messages with identifiable content
        sendChatMessage('Message 1');
        sendChatMessage('Message 2');
        sendChatMessage('Message 3');
        
        setTimeout(() => {
            const outputEvents = receivedEvents.filter(
                evt => evt.nodeId === 'output_1' && evt.type === 'output'
            );
            
            // Check if messages are processed in order
            for (let i = 0; i < outputEvents.length; i++) {
                expect(outputEvents[i].content).to.include(`Message ${i + 1}`);
            }
            
            done();
        }, 20000);
    });
}); 