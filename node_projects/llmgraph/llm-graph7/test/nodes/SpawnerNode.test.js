import WebSocket from 'ws';

describe('SpawnerNode', () => {
    let ws;
    const WS_URL = 'ws://localhost:3000/ws';
    const SPAWNER_ID = 'test_spawner_1';
    
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

    test('should spawn LLM node with prompt data', (done) => {
        const promptData = {
            role: "Test Generator",
            prompt: "Generate a test event"
        };

        const spawnRequest = {
            type: 'spawnNode',
            nodeId: SPAWNER_ID,
            nodeType: 'llm',
            timeout: 30000,
            promptData
        };

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'nodeSpawned' && message.parentNodeId === SPAWNER_ID) {
                expect(message.nodeType).toBe('llm');
                expect(message.promptData).toEqual(promptData);
                expect(message.nodeData.data.label).toBe(promptData.role);
                done();
            }
        });

        ws.send(JSON.stringify(spawnRequest));
    });

    test('should cleanup spawned node after timeout', (done) => {
        const TIMEOUT = 2000; // 2 seconds for testing
        const promptData = {
            role: "Quick Test",
            prompt: "Quick test event"
        };

        const spawnRequest = {
            type: 'spawnNode',
            nodeId: SPAWNER_ID,
            nodeType: 'llm',
            timeout: TIMEOUT,
            promptData
        };

        let spawnedNodeId;
        let spawnTime;

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'nodeSpawned' && message.parentNodeId === SPAWNER_ID) {
                spawnedNodeId = message.spawnedNodeId;
                spawnTime = Date.now();
            }
            if (message.type === 'nodeRemoved' && message.nodeId === spawnedNodeId) {
                const timeElapsed = Date.now() - spawnTime;
                expect(timeElapsed).toBeGreaterThanOrEqual(TIMEOUT);
                done();
            }
        });

        ws.send(JSON.stringify(spawnRequest));
    }, 3000); // 3s timeout

    test('should spawn multiple nodes with correct positioning', (done) => {
        const spawnedNodes = [];
        const SPAWN_COUNT = 3;

        const spawnRequest = {
            type: 'spawnNode',
            nodeId: SPAWNER_ID,
            nodeType: 'llm',
            timeout: 30000,
            promptData: {
                role: "Test Generator",
                prompt: "Generate a test event"
            }
        };

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'nodeSpawned' && message.parentNodeId === SPAWNER_ID) {
                spawnedNodes.push(message.nodeData);
                
                if (spawnedNodes.length === SPAWN_COUNT) {
                    // Check that nodes are positioned correctly relative to each other
                    for (let i = 1; i < spawnedNodes.length; i++) {
                        expect(spawnedNodes[i].position.x).toBe(spawnedNodes[0].position.x);
                        expect(spawnedNodes[i].position.y).toBeGreaterThan(spawnedNodes[i-1].position.y);
                    }
                    done();
                }
            }
        });

        // Spawn multiple nodes
        for (let i = 0; i < SPAWN_COUNT; i++) {
            setTimeout(() => {
                ws.send(JSON.stringify(spawnRequest));
            }, i * 100);
        }
    });
}); 