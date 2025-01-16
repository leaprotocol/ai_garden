import AbstractNode from './AbstractNode.js';

export default class TimerNode extends AbstractNode {
  constructor(id, position) {
    super(id, position);
    this.type = 'timer';
    this.interval = 1000; // Default interval in ms
    this.status = 'stopped';
    this.intervalId = null;
    this.connections = []; // Initialize connections array
  }

  handleEvent(event) {
    switch (event.type) {
      case 'start':
        this.startTimer();
        break;
      case 'stop':
        this.stopTimer();
        break;
      case 'updateInterval':
        this.updateInterval(event.interval);
        break;
      // Handle other event types as needed
      default:
        console.warn(`Unknown event type: ${event.type}`);
    }
  }

  startTimer() {
    if (this.status !== 'running') {
      this.intervalId = setInterval(() => {
        const event = {
          timestamp: Date.now(),
          formattedTime: new Date().toLocaleTimeString(),
          source: this.id,
        };
        this.broadcast({
          type: 'nodeEvent',
          target: this.id,
          payload: event,
        });
      }, this.interval);
      this.updateStatus('running');
    }
  }

  stopTimer() {
    if (this.status === 'running' && this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
      this.updateStatus('stopped');
    }
  }

  updateInterval(newInterval) {
    this.interval = newInterval;
    if (this.status === 'running') {
      this.stopTimer();
      this.startTimer();
    }
    this.broadcast({
      type: 'nodeStatusUpdate',
      nodeId: this.id,
      status: this.status,
    });
  }
} 