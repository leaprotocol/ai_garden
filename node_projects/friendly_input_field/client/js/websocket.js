export class WebSocketClient {
    constructor(responseAreaId = 'responseArea') {
        this.responseAreaId = responseAreaId;
        this.connect();
        this.messageHandlers = new Map();
        this.requestTimings = new Map();
        this.responseContainers = new Map();
        this.ensureResponseArea();
    }

    ensureResponseArea() {
        let responseArea = document.getElementById(this.responseAreaId);
        if (!responseArea) {
            responseArea = document.createElement('div');
            responseArea.id = this.responseAreaId;
            document.body.appendChild(responseArea);
            console.log(`Created responseArea element with ID: ${this.responseAreaId}`);
        } else {
            console.log(`responseArea with ID: ${this.responseAreaId} already exists`);
        }
        return responseArea;
    }

    connect() {
        this.ws = new WebSocket('ws://localhost:3000');
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('Connected');
            this.retryCount = 0;
            this.fetchModels();
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateStatus('Disconnected');
            if (this.retryCount < this.maxRetries) {
                setTimeout(() => {
                    this.retryCount++;
                    this.connect();
                }, 2000);
            }
        };

        this.ws.onmessage = (event) => this.handleMessage(event);

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Error');
        };
    }

    updateStatus(status) {
        const wsStatus = document.getElementById('wsStatus');
        if (wsStatus) {
            wsStatus.textContent = status;
            console.log(`WebSocket status updated to: ${status}`);
        } else {
            console.warn('wsStatus element not found');
        }
    }

    fetchModels() {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'getModels' }));
            console.log('Fetching models...');
        } else {
            console.warn('WebSocket not open. Cannot fetch models.');
        }
    }

    onMessage(type, callback) {
        this.messageHandlers.set(type, callback);
        console.log(`Handler registered for message type: ${type}`);
    }

    send(data) {
        if (this.ws.readyState === WebSocket.OPEN) {
            if (data.requestId) {
                const now = performance.now();
                this.requestTimings.set(data.requestId, {
                    requestStart: now,
                    chunks: 0,
                    firstChunkTime: null,
                    lastChunkTime: null
                });
                console.log(`Request timing initialized for ${data.requestId} at ${now.toFixed(3)}ms`);
            }
            
            this.ws.send(JSON.stringify(data));
            console.log('Sent data to server:', data);
        } else {
            console.error('WebSocket is not open. Current state:', this.ws.readyState);
        }
    }

    handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('Received message:', data);
            
            if (data.requestId) {
                let timing = this.requestTimings.get(data.requestId);
                if (!timing) {
                    console.warn(`No timing data found for ${data.requestId}, creating new timing record`);
                    timing = {
                        requestStart: performance.now(),
                        chunks: 0
                    };
                    this.requestTimings.set(data.requestId, timing);
                }

                if (data.type === 'llm_response') {
                    timing.chunks++;
                    
                    // Fix: Only set firstChunkTime if it hasn't been set yet
                    if (!timing.firstChunkTime) {
                        timing.firstChunkTime = performance.now();
                        console.log(`First chunk received for ${data.requestId} at ${new Date(timing.firstChunkTime).toISOString()}`);
                    }
                    
                    timing.lastChunkTime = Date.now();
                    console.log(`Chunk ${timing.chunks} received at ${new Date(timing.lastChunkTime).toISOString()}`);
                }

                if (data.type === 'llm_complete') {
                    timing.completeTime = Date.now();
                    console.log(`Request ${data.requestId} completed at ${new Date(timing.completeTime).toISOString()}`);
                    
                    // Calculate and log timing metrics
                    const metrics = {
                        totalTime: timing.completeTime - timing.requestStart,
                        timeToFirstChunk: timing.firstChunkTime - timing.requestStart,
                        streamingTime: timing.completeTime - timing.firstChunkTime,
                        totalChunks: timing.chunks
                    };
                    
                    console.log('Final timing metrics:', {
                        requestId: data.requestId,
                        ...metrics,
                        timestamps: {
                            start: new Date(timing.requestStart).toISOString(),
                            firstChunk: new Date(timing.firstChunkTime).toISOString(),
                            complete: new Date(timing.completeTime).toISOString()
                        }
                    });
                    
                    this.displayTimingStats(data.requestId);
                }
            }

            const handler = this.messageHandlers.get(data.type);
            if (handler) {
                handler(data);
            }
        } catch (error) {
            console.error('Error handling message:', error);
            console.debug('Raw message:', event.data);
        }
    }

    appendResponse(text, requestId) {
        console.log(`Appending response for ${requestId}`);
        
        const responseArea = this.ensureResponseArea();
        let containerData = this.responseContainers.get(requestId);

        // Clear any existing direct response divs
        const oldResponses = responseArea.querySelectorAll('.response:not(.response-container)');
        oldResponses.forEach(el => el.remove());

        if (!containerData) {
            // Create new container structure
            const container = document.createElement('div');
            container.className = 'response-container';
            container.dataset.requestId = requestId;

            const textDiv = document.createElement('div');
            textDiv.className = 'response-text';
            container.appendChild(textDiv);

            const statsDiv = document.createElement('div');
            statsDiv.className = 'response-stats';
            container.appendChild(statsDiv);

            // Insert at the top of response area
            if (responseArea.firstChild) {
                responseArea.insertBefore(container, responseArea.firstChild);
            } else {
                responseArea.appendChild(container);
            }

            containerData = { container, textDiv, statsDiv };
            this.responseContainers.set(requestId, containerData);
            
            console.log(`Created new container for ${requestId}, current containers:`, 
                Array.from(this.responseContainers.keys()));
        }

        // Append text to the text div
        containerData.textDiv.textContent += text;
        
        // Scroll to the latest content
        responseArea.scrollTop = responseArea.scrollHeight;
        
        return containerData;
    }

    displayTimingStats(requestId) {
        console.log('Displaying timing stats for request:', requestId);
        
        const timing = this.requestTimings.get(requestId);
        if (!timing) {
            console.error('No timing data found for request:', requestId);
            return;
        }

        const containerData = this.responseContainers.get(requestId);
        if (!containerData?.statsDiv) {
            console.error('No container or stats div found for request:', requestId);
            return;
        }

        // Ensure all required timestamps exist
        if (!timing.requestStart || !timing.firstChunkTime || !timing.completeTime) {
            console.error('Missing required timestamps:', {
                requestStart: timing.requestStart,
                firstChunkTime: timing.firstChunkTime,
                completeTime: timing.completeTime
            });
            return;
        }

        const stats = {
            totalTime: timing.completeTime - timing.requestStart,
            timeToFirstChunk: timing.firstChunkTime - timing.requestStart,
            streamingTime: timing.completeTime - timing.firstChunkTime,
            totalChunks: timing.chunks
        };

        // Format timestamps in local timezone
        const formatTime = (timestamp) => {
            return new Date(timestamp).toLocaleTimeString(undefined, {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                fractionalSecondDigits: 3
            });
        };

        containerData.statsDiv.innerHTML = `
            <div class="timing-stats">
                <div class="stat">Total Time: ${stats.totalTime.toFixed(1)}ms</div>
                <div class="stat">Time to First Chunk: ${stats.timeToFirstChunk.toFixed(1)}ms</div>
                <div class="stat">Streaming Time: ${stats.streamingTime.toFixed(1)}ms</div>
                <div class="stat">Total Chunks: ${stats.totalChunks}</div>
                <div class="stat">Request Start: ${formatTime(timing.requestStart)}</div>
                <div class="stat">First Chunk: ${formatTime(timing.firstChunkTime)}</div>
                <div class="stat">Complete: ${formatTime(timing.completeTime)}</div>
            </div>
        `;
        
        console.log('Timing details:', {
            requestId,
            stats,
            timestamps: {
                requestStart: new Date(timing.requestStart).toISOString(),
                firstChunkTime: new Date(timing.firstChunkTime).toISOString(),
                completeTime: new Date(timing.completeTime).toISOString()
            }
        });
    }

    displayTiming(requestId) {
        const timing = this.requestTimings.get(requestId);
        if (!timing) return;

        const now = Date.now();
        const timeToFirstChunk = timing.firstChunkTime ? (timing.firstChunkTime - timing.requestStart) : null;
        const totalTime = timing.lastChunkTime ? (timing.lastChunkTime - timing.requestStart) : (now - timing.requestStart);
        const streamingTime = timing.lastChunkTime ? (timing.lastChunkTime - timing.firstChunkTime) : null;

        const formatTime = (time) => time ? `${time.toFixed(1)}ms` : 'N/A';
        const formatTimestamp = (timestamp) => timestamp ? new Date(timestamp).toLocaleTimeString('en-US', {
            hour12: false,
            fractionalSecondDigits: 3
        }) : 'N/A';

        const timingHtml = `
            <div class="timing-stats">
                <div class="stat">Total Time: ${formatTime(totalTime)}</div>
                <div class="stat">Time to First Chunk: ${formatTime(timeToFirstChunk)}</div>
                <div class="stat">Streaming Time: ${formatTime(streamingTime)}</div>
                <div class="stat">Total Chunks: ${timing.chunks}</div>
                <div class="stat">Request Start: ${formatTimestamp(timing.requestStart)}</div>
                <div class="stat">First Chunk: ${formatTimestamp(timing.firstChunkTime)}</div>
                <div class="stat">Complete: ${formatTimestamp(timing.lastChunkTime)}</div>
            </div>
        `;

        // Log detailed timing information for debugging
        console.log('Timing details:', {
            requestId,
            requestStart: timing.requestStart,
            firstChunkTime: timing.firstChunkTime,
            lastChunkTime: timing.lastChunkTime,
            timeToFirstChunk,
            totalTime,
            streamingTime,
            chunks: timing.chunks
        });

        return timingHtml;
    }
}