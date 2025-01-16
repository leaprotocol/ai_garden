import Node from './Node.js';

class TimerNode extends Node {
    constructor(id, type, config = {}) {
        super(id, type, config);
        this.interval = config.interval || 1000;
        this.timer = null;
    }

    start() {
        if (!this.timer) {
            this.timer = setInterval(() => {
                // Timer logic here
                console.log(`Timer Node ${this.id} triggered at interval ${this.interval}`);
            }, this.interval);
            this.running = true;
        }
    }

    stop() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
            this.running = false;
        }
    }

    // Additional TimerNode-specific methods
}

export default TimerNode; 