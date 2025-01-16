import AbstractNode from './AbstractNode.js';

export default class OutputNode extends AbstractNode {
  constructor(id, position) {
    super(id, position);
    this.type = 'output';
    this.log = [];
    this.connections = []; // Initialize connections array
  }

  handleEvent(event) {
    switch (event.type) {
      case 'logEvent':
        this.logEvent(event.payload);
        break;
      // Handle other event types as needed
      default:
        console.warn(`Unknown event type: ${event.type}`);
    }
  }

  logEvent(payload) {
    this.log.push(payload);
    this.broadcast({
      type: 'nodeLogUpdate',
      nodeId: this.id,
      log: this.log,
    });
  }
} 