import WebSocket from 'ws';

describe('LeakyBucketNode', () => {
    let ws;
    const WS_URL = 'ws://localhost:3000/ws';
    const BUCKET_ID = 'test_bucket_1';
    
    beforeEach((done) => {
        ws = new WebSocket(WS_URL);
        ws.on('open', () => {
            // Create a fresh bucket for each test
            createBucketNode();
            done();
        });
    });

    afterEach((done) => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
        done();
    });

    function createBucketNode() {
        const bucketNode = {
            type: 'createNode',
            nodeType: 'leakyBucket',
            nodeId: BUCKET_ID,
            data: {
                label: 'Test Bucket',
                bucketSize: 5,
                leakInterval: 1000 // 1 second for faster testing
            }
        };
        ws.send(JSON.stringify(bucketNode));
    }

    test('should create bucket node successfully', (done) => {
        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'nodeCreated' && message.nodeId === BUCKET_ID) {
                expect(message.nodeType).toBe('leakyBucket');
                expect(message.data.bucketSize).toBe(5);
                expect(message.data.leakInterval).toBe(1000);
                done();
            }
        });
    });

    test('should receive and buffer events', (done) => {
        const events = [];
        const testEvent = {
            type: 'nodeOutput',
            nodeId: 'test_source',
            targetId: BUCKET_ID,
            content: 'Test event content'
        };

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'bucketEvent') {
                events.push(message);
                if (events.length === 1) {
                    expect(message.content).toBe('Test event content');
                    done();
                }
            }
        });

        // Send test event
        ws.send(JSON.stringify(testEvent));
    });

    test('should leak events at specified interval', (done) => {
        const events = [];
        const testEvents = [
            'Event 1',
            'Event 2',
            'Event 3'
        ].map(content => ({
            type: 'nodeOutput',
            nodeId: 'test_source',
            targetId: BUCKET_ID,
            content
        }));

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'bucketLeaked') {
                events.push(message);
                if (events.length === testEvents.length) {
                    expect(events.map(e => e.content)).toEqual(
                        expect.arrayContaining(testEvents.map(e => e.content))
                    );
                    done();
                }
            }
        });

        // Send test events in sequence
        testEvents.forEach((event, i) => {
            setTimeout(() => {
                ws.send(JSON.stringify(event));
            }, i * 200); // Send every 200ms
        });
    }, 5000); // 5s timeout for this test
}); 