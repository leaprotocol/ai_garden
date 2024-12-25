export default class AbstractNode {
  constructor(id, position) {
    this.id = id;
    this.position = position;
    this.type = 'abstract';
    this.status = 'initialized';
    this.connections = [];
  }

  initialize(broadcast) {
    this.broadcast = broadcast;
  }

  handleEvent(event) {
    // To be implemented by subclasses
  }

  updateStatus(newStatus) {
    this.status = newStatus;
    this.broadcast({
      type: 'nodeStatusUpdate',
      nodeId: this.id,
      status: this.status,
    });
  }

  addConnection(targetNodeId) {
    if (!this.connections.includes(targetNodeId)) {
      this.connections.push(targetNodeId);
      this.broadcast({
        type: 'connectionAdded',
        sourceNodeId: this.id,
        targetNodeId: targetNodeId,
      });
    }
  }

  removeConnection(targetNodeId) {
    this.connections = this.connections.filter(id => id !== targetNodeId);
    this.broadcast({
      type: 'connectionRemoved',
      sourceNodeId: this.id,
      targetNodeId: targetNodeId,
    });
  }

  getConnections() {
    return this.connections;
  }
} 