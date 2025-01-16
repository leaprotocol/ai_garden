export class EventCapture {
    constructor(wsClient) {
        this.wsClient = wsClient;
        this.eventQueue = [];
        this.requestTimings = new Map(); // Track request timings
        this.setupEventListeners();
        this.setupQueueControls();
    }

    setupQueueControls() {
        const processBtn = document.getElementById('processEvents');
        const clearBtn = document.getElementById('clearEvents');
        
        processBtn?.addEventListener('click', () => this.processEventQueue());
        clearBtn?.addEventListener('click', () => this.clearEventQueue());
    }

    setupEventListeners() {
        const input = document.getElementById('userInput');
        const events = ['keydown', 'keyup', 'mousedown', 'mousemove'];
        
        events.forEach(type => {
            input?.addEventListener(type, e => this.handleEvent(e));
        });
    }

    handleEvent(event) {
        const eventData = {
            type: event.type,
            timestamp: Date.now(),
            data: this.extractEventData(event)
        };

        this.eventQueue.push(eventData);
        this.updateUI(eventData);
    }

    extractEventData(event) {
        const base = {
            type: event.type,
            target: event.target.id
        };

        if (event.type.startsWith('key')) {
            base.key = event.key;
            base.code = event.code;
        } else if (event.type.startsWith('mouse')) {
            base.x = event.clientX;
            base.y = event.clientY;
        }

        return base;
    }

    updateUI(eventData) {
        // Update event counter
        const countElement = document.getElementById('eventCount');
        if (countElement) {
            countElement.textContent = this.eventQueue.length.toString();
        }

        // Add event to history
        const eventList = document.getElementById('eventList');
        if (eventList) {
            const eventElement = document.createElement('div');
            eventElement.className = 'event';
            eventElement.textContent = JSON.stringify(eventData, null, 2);
            eventList.appendChild(eventElement);
            eventList.scrollTop = eventList.scrollHeight;
        }
    }

    processEventQueue() {
        if (this.eventQueue.length === 0) {
            console.log('Event queue is empty');
            return;
        }

        const requestId = crypto.randomUUID();
        const timing = {
            requestStart: performance.now(),
            requestId: requestId
        };
        this.requestTimings.set(requestId, timing);

        const message = {
            type: 'event_batch',
            events: this.eventQueue,
            model: document.getElementById('modelSelect')?.value,
            requestId: requestId
        };

        debugLog('Sending message to server', {
            ...message,
            timing: timing.requestStart
        });
        
        this.wsClient.send(message);
        this.clearEventQueue();
    }

    clearEventQueue() {
        this.eventQueue = [];
        this.updateUI(null);
        
        // Clear event history
        const eventList = document.getElementById('eventList');
        if (eventList) {
            eventList.innerHTML = '';
        }
        
        // Reset counter
        const countElement = document.getElementById('eventCount');
        if (countElement) {
            countElement.textContent = '0';
        }
    }
} 