import { expect } from 'chai';
import WebSocket from 'ws';

const WS_URL = 'ws://localhost:3000/ws';
const SPAWNER_ID = 'test_spawner_1';

describe('Spawner Node', () => {
    let ws;
    let spawnedNodes = new Set();

    before((done) => {
        ws = new WebSocket(WS_URL);
        ws.on('open', () => {
            console.log('Connected to WebSocket server');
            createSpawnerNode();
            done();
        });

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'nodeSpawned' && message.parentNodeId === SPAWNER_ID) {
                spawnedNodes.add(message.spawnedNodeId);
            }
            if (message.type === 'nodeRemoved') {
                spawnedNodes.delete(message.nodeId);
            }
        });
    });

    after(() => {
        ws.close();
    });

    beforeEach(() => {
        spawnedNodes.clear();
    });

    function createSpawnerNode() {
        const spawnerNode = {
            type: 'createNode',
            nodeType: 'spawner',
            nodeId: SPAWNER_ID,
            data: {
                label: 'Test Spawner'
            }
        };
        ws.send(JSON.stringify(spawnerNode));
    }

    function requestSpawn(nodeType = 'llm', timeout = 5000, promptData = null) {
        const spawnRequest = {
            type: 'spawnNode',
            nodeId: SPAWNER_ID,
            nodeType,
            timeout,
            promptData,
            timestamp: Date.now()
        };
        ws.send(JSON.stringify(spawnRequest));
    }

    it('should create a spawner node', (done) => {
        setTimeout(() => {
            expect(ws.readyState).to.equal(WebSocket.OPEN);
            done();
        }, 1000);
    });

    it('should spawn an LLM node', (done) => {
        requestSpawn('llm', 5000, {
            role: 'test',
            prompt: 'Test prompt'
        });

        setTimeout(() => {
            expect(spawnedNodes.size).to.equal(1);
            const nodeId = Array.from(spawnedNodes)[0];
            expect(nodeId).to.include('spawned_');
            done();
        }, 2000);
    });

    it('should handle multiple spawns', (done) => {
        for (let i = 0; i < 3; i++) {
            requestSpawn('llm', 5000, {
                role: `test_${i}`,
                prompt: `Test prompt ${i}`
            });
        }

        setTimeout(() => {
            expect(spawnedNodes.size).to.equal(3);
            done();
        }, 2000);
    });

    it('should cleanup spawned nodes after timeout', (done) => {
        const shortTimeout = 2000;
        requestSpawn('llm', shortTimeout, {
            role: 'temporary',
            prompt: 'This node should be cleaned up'
        });

        // Check that node exists initially
        setTimeout(() => {
            expect(spawnedNodes.size).to.equal(1);
            
            // Then check that it's cleaned up after timeout
            setTimeout(() => {
                expect(spawnedNodes.size).to.equal(0);
                done();
            }, shortTimeout + 1000);
        }, 1000);
    });

    it('should handle different node types', (done) => {
        // Test spawning different types of nodes
        requestSpawn('timer', 5000);
        requestSpawn('llm', 5000);
        
        setTimeout(() => {
            expect(spawnedNodes.size).to.equal(2);
            done();
        }, 2000);
    });

    it('should position spawned nodes correctly', (done) => {
        let nodePositions = [];
        
        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'nodeSpawned' && message.parentNodeId === SPAWNER_ID) {
                nodePositions.push(message.nodeData.position);
            }
        });

        // Spawn multiple nodes
        for (let i = 0; i < 3; i++) {
            requestSpawn('llm', 5000);
        }

        setTimeout(() => {
            expect(nodePositions.length).to.equal(3);
            // Check that nodes are positioned at different coordinates
            for (let i = 1; i < nodePositions.length; i++) {
                expect(nodePositions[i].y).to.not.equal(nodePositions[i-1].y);
            }
            done();
        }, 2000);
    });
}); 