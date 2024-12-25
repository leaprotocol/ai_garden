import { expect } from 'chai';
import WebSocket from 'ws';

const WS_URL = 'ws://localhost:3000/ws';

describe('News Summarization Scenario', () => {
    let ws;
    let nodes = {};
    let receivedEvents = [];
    let isConnected = false;

    before(function(done) {
        this.timeout(10000); // Increase timeout for setup
        
        ws = new WebSocket(WS_URL);
        
        ws.on('open', () => {
            console.log('WebSocket connected');
            isConnected = true;
            setupNodes();
            done();
        });

        ws.on('error', (error) => {
            console.error('WebSocket error:', error);
        });

        ws.on('close', () => {
            console.log('WebSocket closed');
            isConnected = false;
        });

        ws.on('message', (data) => {
            try {
                const message = JSON.parse(data);
                // Log raw message for debugging
                console.log('Raw message:', JSON.stringify(message, null, 2));
                
                receivedEvents.push(message);
                
                if (message.type === 'nodeCreated') {
                    console.log('Node created:', message.nodeId, message.nodeType);
                    nodes[message.nodeId] = message.nodeType;
                } else if (message.type === 'error') {
                    console.error('Server error:', message.error);
                }
                
                // Log specific events we're interested in
                if (message.type === 'bucketUpdate') {
                    console.log('Bucket update:', message.nodeId, 'count:', message.count);
                }
                if (message.type === 'llmResponse' || message.type === 'llmResponseChunk') {
                    console.log('LLM response:', message.nodeId, 'content length:', message.content?.length);
                }
            } catch (error) {
                console.error('Error parsing message:', error, 'Raw data:', data.toString());
            }
        });
    });

    after(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
    });

    beforeEach(() => {
        receivedEvents = [];
    });

    function setupNodes() {
        console.log('Setting up nodes...');
        
        const nodes = [
            {
                type: 'createNode',
                nodeType: 'timer',
                nodeId: 'timer_1',
                data: {
                    label: 'News Timer',
                    interval: 5000 // 5 seconds for testing
                }
            },
            {
                type: 'createNode',
                nodeType: 'llm',
                nodeId: 'llm_fetch',
                data: {
                    label: 'News Fetcher',
                    model: 'smollm:latest',
                    systemPrompt: 'You are a technology news reporter. Generate short, factual updates.',
                    prompt: 'Generate a short news update about technology:'
                }
            },
            {
                type: 'createNode',
                nodeType: 'leakyBucket',
                nodeId: 'bucket_1',
                data: {
                    label: 'News Accumulator',
                    bucketSize: 3,
                    leakInterval: 10000
                }
            },
            {
                type: 'createNode',
                nodeType: 'llm',
                nodeId: 'llm_summarize',
                data: {
                    label: 'News Summarizer',
                    model: 'smollm:latest',
                    systemPrompt: 'You are a news editor. Summarize multiple news items into a concise update.',
                    prompt: 'Summarize the following news updates:'
                }
            },
            {
                type: 'createNode',
                nodeType: 'output',
                nodeId: 'output_1',
                data: {
                    label: 'News Output'
                }
            }
        ];

        // Send node creation requests sequentially with acknowledgment
        function createNextNode(index) {
            if (index >= nodes.length) {
                console.log('All nodes created, waiting for confirmations...');
                return;
            }

            const node = nodes[index];
            console.log(`Creating node ${index + 1}/${nodes.length}:`, node.nodeId);
            
            // Send create request
            ws.send(JSON.stringify(node));

            // Wait for node creation confirmation or timeout
            let confirmed = false;
            const confirmTimeout = setTimeout(() => {
                if (!confirmed) {
                    console.error(`Timeout waiting for node creation confirmation: ${node.nodeId}`);
                    createNextNode(index + 1);
                }
            }, 5000);

            // Setup one-time handler for this node's creation confirmation
            const confirmHandler = (data) => {
                try {
                    const message = JSON.parse(data);
                    if (message.type === 'nodeCreated' && message.nodeId === node.nodeId) {
                        confirmed = true;
                        clearTimeout(confirmTimeout);
                        console.log(`Node creation confirmed: ${node.nodeId}`);
                        setTimeout(() => createNextNode(index + 1), 500);
                    }
                } catch (error) {
                    console.error('Error in confirm handler:', error);
                }
            };

            ws.on('message', confirmHandler);
        }

        // Start creating nodes
        createNextNode(0);

        // Wait for all nodes to be created before connecting
        const checkInterval = setInterval(() => {
            const createdCount = Object.keys(nodes).length;
            console.log(`Checking node creation progress: ${createdCount}/${nodes.length}`);
            
            if (createdCount === nodes.length) {
                clearInterval(checkInterval);
                console.log('All nodes confirmed, connecting...');
                connectNodes();
            }
        }, 1000);
    }

    function connectNodes() {
        console.log('Connecting nodes...');
        const connections = [
            {
                type: 'connect',
                sourceId: 'timer_1',
                targetId: 'llm_fetch'
            },
            {
                type: 'connect',
                sourceId: 'llm_fetch',
                targetId: 'bucket_1'
            },
            {
                type: 'connect',
                sourceId: 'bucket_1',
                targetId: 'llm_summarize'
            },
            {
                type: 'connect',
                sourceId: 'llm_summarize',
                targetId: 'output_1'
            }
        ];

        // Send connection requests sequentially
        let promise = Promise.resolve();
        connections.forEach((conn, index) => {
            promise = promise.then(() => {
                return new Promise(resolve => {
                    console.log(`Creating connection ${index + 1}/${connections.length}`);
                    ws.send(JSON.stringify(conn));
                    setTimeout(resolve, 500); // Wait between connections
                });
            });
        });
    }

    it('should create all required nodes', function(done) {
        this.timeout(10000); // Increase timeout
        
        function checkNodes() {
            if (Object.keys(nodes).length === 5) {
                console.log('All nodes created:', nodes);
                expect(nodes['timer_1']).to.equal('timer');
                expect(nodes['llm_fetch']).to.equal('llm');
                expect(nodes['bucket_1']).to.equal('leakyBucket');
                expect(nodes['llm_summarize']).to.equal('llm');
                expect(nodes['output_1']).to.equal('output');
                done();
            } else {
                console.log('Waiting for nodes, current count:', Object.keys(nodes).length);
                setTimeout(checkNodes, 1000);
            }
        }

        setTimeout(checkNodes, 1000);
    });

    it('should process timer events through the pipeline', function(done) {
        this.timeout(40000); // Increase timeout for full pipeline
        
        console.log('Starting timer test...');
        ws.send(JSON.stringify({
            type: 'timerStart',
            nodeId: 'timer_1'
        }));

        let checkCount = 0;
        function checkOutput() {
            const outputEvents = receivedEvents.filter(
                evt => evt.nodeId === 'output_1' && evt.type === 'output'
            );
            
            console.log('Checking output events, count:', outputEvents.length);
            
            if (outputEvents.length > 0) {
                const lastOutput = outputEvents[outputEvents.length - 1];
                expect(lastOutput.content).to.be.a('string');
                expect(lastOutput.content.length).to.be.greaterThan(0);
                
                // Stop the timer
                ws.send(JSON.stringify({
                    type: 'timerStop',
                    nodeId: 'timer_1'
                }));
                
                done();
            } else if (checkCount < 6) { // Check for 30 seconds
                checkCount++;
                setTimeout(checkOutput, 5000);
            } else {
                done(new Error('No output events received after 30 seconds'));
            }
        }

        setTimeout(checkOutput, 5000);
    });

    it('should accumulate content in LeakyBucket', function(done) {
        this.timeout(20000);
        
        function checkBucket() {
            const bucketEvents = receivedEvents.filter(
                evt => evt.nodeId === 'bucket_1' && evt.type === 'bucketUpdate'
            );
            
            console.log('Checking bucket events, count:', bucketEvents.length);
            
            if (bucketEvents.length > 0) {
                const lastBucketEvent = bucketEvents[bucketEvents.length - 1];
                expect(lastBucketEvent.count).to.be.lessThanOrEqual(3);
                done();
            } else {
                setTimeout(checkBucket, 5000);
            }
        }

        checkBucket();
    });

    it('should generate summaries from accumulated content', function(done) {
        this.timeout(20000);
        
        function checkSummaries() {
            const summaryEvents = receivedEvents.filter(
                evt => evt.nodeId === 'llm_summarize' && 
                      (evt.type === 'llmResponse' || evt.type === 'llmResponseChunk')
            );
            
            console.log('Checking summary events, count:', summaryEvents.length);
            
            if (summaryEvents.length > 0) {
                const lastSummary = summaryEvents[summaryEvents.length - 1];
                expect(lastSummary.content || lastSummary.chunk).to.be.a('string');
                expect((lastSummary.content || lastSummary.chunk).length).to.be.greaterThan(0);
                done();
            } else {
                setTimeout(checkSummaries, 5000);
            }
        }

        checkSummaries();
    });
}); 