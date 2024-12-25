import { AbstractNode } from './AbstractNode.js';

export class OutputNode extends AbstractNode {
  constructor() {
    super();
    this.type = 'output';
    this.events = [];
  }

  receive(event) {
    this.events.push(event);
    this.emit('eventReceived', this.id, event);
  }

  clearEvents() {
    this.events = [];
    this.emit('eventsCleared', this.id);
  }
} 