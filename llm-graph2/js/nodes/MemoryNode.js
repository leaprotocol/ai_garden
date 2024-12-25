import Node from './Node.js';

class MemoryNode extends Node {
    constructor(id, type, config = {}) {
        super(id, type, config);
        this.memoryType = config.memoryType || 'short';
        this.maxSize = config.maxSize || 10;
        this.retrievalStrategy = config.retrievalStrategy || 'fifo';
        this.entries = [];
    }

    addEntry(entry) {
        if (this.entries.length >= this.maxSize) {
            if (this.retrievalStrategy === 'fifo') {
                this.entries.shift();
            }
            // Implement other strategies if needed
        }
        this.entries.push(entry);
    }

    clearMemory() {
        this.entries = [];
    }

    // Additional MemoryNode-specific methods
}

export default MemoryNode; 