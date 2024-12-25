import { EventEmitter } from 'events';
import crypto from 'crypto';

export class AbstractNode extends EventEmitter {
  constructor() {
    super();
    this.id = crypto.randomUUID();
    this.type = 'abstract';
    this.status = 'idle';
  }

  setStatus(status) {
    this.status = status;
    this.emit('statusChanged', this.id, this.status);
  }

  // Implement common functionalities here
} 