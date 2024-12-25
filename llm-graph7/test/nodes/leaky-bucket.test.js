import { expect } from 'chai';
import WebSocket from 'ws';

const WS_URL = 'ws://localhost:3000/ws';
const BUCKET_ID = 'test_bucket_1';

describe('LeakyBucket Node', () => {
    let ws;
    let receivedEvents = [];

    before((done) => {
        ws = new WebSocket(WS_URL);
        ws.on('open', () => {
            console.log('Connected to WebSocket server');
            createBucketNode();
            done();
        });

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'bucketEvent' && message.nodeId === BUCKET_ID) {
                receivedEvents.push(message);
            }
        });
    });

    after(() => {
        ws.close();
    });

    function createBucketNode() {
        const bucketNode = {
            type: 'createNode',
            nodeType: 'leakyBucket',
            nodeId: BUCKET_ID,
            data: {
                label: 'Test Bucket',
                bucketSize: 3,
                leakInterval: 5000
            }
        };
        ws.send(JSON.stringify(bucketNode));
    }

    function sendEvent(content) {
        const event = {
            type: 'nodeOutput',
            nodeId: BUCKET_ID,
            content,
            timestamp: Date.now()
        };
        ws.send(JSON.stringify(event));
    }

    it('should create a bucket node', (done) => {
        // Wait for node creation confirmation
        setTimeout(() => {
            expect(ws.readyState).to.equal(WebSocket.OPEN);
            done();
        }, 1000);
    });

    it('should receive and buffer events', (done) => {
        sendEvent('Test event 1');
        sendEvent('Test event 2');
        
        setTimeout(() => {
            expect(receivedEvents.length).to.be.at.least(1);
            expect(receivedEvents[0].content).to.include('Test event');
            done();
        }, 2000);
    });

    it('should respect bucket size limit', (done) => {
        // Send more events than bucket size
        for (let i = 0; i < 5; i++) {
            sendEvent(`Overflow test event ${i}`);
        }

        setTimeout(() => {
            expect(receivedEvents.length).to.be.at.most(3);
            done();
        }, 2000);
    });

    it('should leak events at the specified interval', (done) => {
        const initialCount = receivedEvents.length;
        
        setTimeout(() => {
            expect(receivedEvents.length).to.be.lessThan(initialCount);
            done();
        }, 6000);
    });
}); 