class Node {
    constructor(id, type, config = {}) {
        this.id = id;
        this.type = type;
        this.config = config;
        this.connections = [];
        this.running = false;
    }

    connect(targetNode) {
        this.connections.push(targetNode);
    }

    disconnect(targetNode) {
        this.connections = this.connections.filter(node => node.id !== targetNode.id);
    }

    emit(event) {
        this.connections.forEach(node => node.handleEvent(event));
    }

    handleEvent(event) {
        if (!this.running) return;
        this.process(event);
    }

    start() {
        this.running = true;
    }

    stop() {
        this.running = false;
    }

    process(event) {
        throw new Error('Process method must be implemented by subclass.');
    }
}

export default Node; 