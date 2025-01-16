import { AbstractNode } from './AbstractNode.js';

export class TimerNode extends AbstractNode {
  constructor(config = {}) {
    super();
    this.type = 'timer';
    this.interval = config.interval || 1000;
    this.isRunning = false;
    this.timer = null;
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.setStatus('running');
    this.timer = setInterval(() => {
      this.emit('tick', Date.now());
    }, this.interval);
  }

  stop() {
    if (!this.isRunning) return;
    clearInterval(this.timer);
    this.timer = null;
    this.isRunning = false;
    this.setStatus('idle');
  }

  setInterval(interval) {
    this.interval = interval;
    if (this.isRunning) {
      this.stop();
      this.start();
    }
  }
} 