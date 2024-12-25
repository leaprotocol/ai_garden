import { expect } from 'chai';
import WebSocket from 'ws';

const WS_URL = 'ws://localhost:3000/ws';

describe('Parallel Processing Scenario', () => {
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
            receivedEvents.push(message);
            
            // Track node creation confirmations
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
        // Create Timer node
        ws.send(JSON.stringify({
            type: 'createNode',
            nodeType: 'timer',
            nodeId: 'timer_1',
            data: {
                label: 'Task Timer',
                interval: 10000 // 10 seconds
            }
        }));

        // Create Spawner node
        ws.send(JSON.stringify({
            type: 'createNode',
            nodeType: 'spawner',
            nodeId: 'spawner_1',
            data: {
                label: 'LLM Spawner',
                maxNodes: 3,
                nodeType: 'llm',
                timeout: 30000,
                nodeConfig: {
                    model: 'smollm:latest',
                    prompt: 'Generate a creative story about: ${input}'
                }
            }
        }));

        // Create LeakyBucket for collecting results
        ws.send(JSON.stringify({
            type: 'createNode',
            nodeType: 'leakyBucket',
            nodeId: 'bucket_1',
            data: {
                label: 'Results Collector',
                bucketSize: 5,
                leakInterval: 15000
            }
        }));

        // Create Output node
        ws.send(JSON.stringify({
            type: 'createNode',
            nodeType: 'output',
            nodeId: 'output_1',
            data: {
                label: 'Final Output'
            }
        }));

        // Connect nodes
        setTimeout(() => {
            connectNodes();
        }, 1000);
    }

    function connectNodes() {
        // Timer -> Spawner
        ws.send(JSON.stringify({
            type: 'connect',
            sourceId: 'timer_1',
            targetId: 'spawner_1'
        }));

        // Spawner's spawned nodes will auto-connect to bucket
        ws.send(JSON.stringify({
            type: 'connect',
            sourceId: 'spawner_1',
            targetId: 'bucket_1',
            isSpawnerConnection: true
        }));

        // Bucket -> Output
        ws.send(JSON.stringify({
            type: 'connect',
            sourceId: 'bucket_1',
            targetId: 'output_1'
        }));
    }

    it('should create base nodes successfully', (done) => {
        setTimeout(() => {
            expect(Object.keys(nodes).length).to.be.at.least(4);
            expect(nodes['timer_1']).to.equal('timer');
            expect(nodes['spawner_1']).to.equal('spawner');
            expect(nodes['bucket_1']).to.equal('leakyBucket');
            expect(nodes['output_1']).to.equal('output');
            done();
        }, 2000);
    });

    it('should spawn LLM nodes on timer events', (done) => {
        // Start the timer
        ws.send(JSON.stringify({
            type: 'timerStart',
            nodeId: 'timer_1'
        }));

        // Wait for spawned nodes
        setTimeout(() => {
            const spawnedNodes = Object.entries(nodes)
                .filter(([id, type]) => id.startsWith('spawned_') && type === 'llm');
            
            expect(spawnedNodes.length).to.be.at.least(1);
            expect(spawnedNodes.length).to.be.at.most(3); // maxNodes limit
            
            // Stop timer
            ws.send(JSON.stringify({
                type: 'timerStop',
                nodeId: 'timer_1'
            }));
            
            done();
        }, 15000);
    });

    it('should process content through spawned nodes', (done) => {
        const llmResponses = receivedEvents.filter(
            evt => evt.type === 'llmResponse' && evt.nodeId.startsWith('spawned_')
        );

        expect(llmResponses.length).to.be.greaterThan(0);
        
        // Verify responses from different nodes
        const uniqueSourceNodes = new Set(llmResponses.map(evt => evt.nodeId));
        expect(uniqueSourceNodes.size).to.be.greaterThan(1);
        
        done();
    });

    it('should collect results in LeakyBucket', (done) => {
        const bucketEvents = receivedEvents.filter(
            evt => evt.nodeId === 'bucket_1' && evt.type === 'bucketUpdate'
        );
        
        expect(bucketEvents.length).to.be.greaterThan(0);
        
        // Check bucket accumulation
        const lastBucketEvent = bucketEvents[bucketEvents.length - 1];
        expect(lastBucketEvent.count).to.be.lessThanOrEqual(5); // Max bucket size
        
        done();
    });

    it('should cleanup spawned nodes after timeout', (done) => {
        // Wait for node cleanup
        setTimeout(() => {
            const removeEvents = receivedEvents.filter(
                evt => evt.type === 'nodeRemoved' && evt.nodeId.startsWith('spawned_')
            );
            
            expect(removeEvents.length).to.be.greaterThan(0);
            done();
        }, 35000); // Wait for timeout + buffer
    });
}); 