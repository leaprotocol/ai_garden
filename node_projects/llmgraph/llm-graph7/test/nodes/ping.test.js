import { expect } from 'chai';
import WebSocket from 'ws';

const WS_URL = 'ws://localhost:3000/ws';

describe('WebSocket Connection Test', () => {
    let ws;

    before(function(done) {
        this.timeout(5000);
        ws = new WebSocket(WS_URL);
        
        ws.on('open', () => {
            console.log('WebSocket connected');
            done();
        });

        ws.on('error', (error) => {
            console.error('WebSocket error:', error);
        });
    });

    after(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
    });

    it('should receive ping response', function(done) {
        this.timeout(5000);
        
        ws.on('message', (data) => {
            try {
                const message = JSON.parse(data);
                console.log('Received message:', message);
                
                if (message.type === 'pong') {
                    expect(message.timestamp).to.be.a('number');
                    done();
                }
            } catch (error) {
                console.error('Error parsing message:', error);
            }
        });

        console.log('Sending ping...');
        ws.send(JSON.stringify({
            type: 'ping',
            timestamp: Date.now()
        }));
    });

    it('should receive available models', function(done) {
        this.timeout(5000);
        
        ws.on('message', (data) => {
            try {
                const message = JSON.parse(data);
                console.log('Received message:', message);
                
                if (message.type === 'availableModels') {
                    expect(message.models).to.be.an('array');
                    done();
                }
            } catch (error) {
                console.error('Error parsing message:', error);
            }
        });

        console.log('Requesting models...');
        ws.send(JSON.stringify({
            type: 'getModels'
        }));
    });
}); 