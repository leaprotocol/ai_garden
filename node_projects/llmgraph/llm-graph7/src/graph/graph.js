import { EventEmitter } from 'events';
import { TimerNode } from './nodes/TimerNode.js';
import { OutputNode } from './nodes/OutputNode.js';

export class Graph extends EventEmitter {
  constructor() {
    super();
    this.nodes = new Map();
    this.connections = [];
    this.nodeTypes = {
      timer: TimerNode,
      output: OutputNode,
      // Add other node types here
    };
  }

  addNode(type, config = {}) {
    const NodeClass = this.nodeTypes[type];
    if (!NodeClass) {
      throw new Error(`Node type "${type}" not found.`);
    }
    const node = new NodeClass(config);
    this.nodes.set(node.id, node);
    this.emit('nodeAdded', node);
    return node;
  }

  connect(sourceNodeId, targetNodeId) {
    const sourceNode = this.nodes.get(sourceNodeId);
    const targetNode = this.nodes.get(targetNodeId);
    if (!sourceNode || !targetNode) {
      throw new Error('Source or target node not found.');
    }
    this.connections.push({ source: sourceNodeId, target: targetNodeId });
    this.emit('connectionAdded', sourceNodeId, targetNodeId);
  }

  removeNode(nodeId) {
    this.nodes.delete(nodeId);
    this.connections = this.connections.filter(
      (conn) => conn.source !== nodeId && conn.target !== nodeId
    );
    this.emit('nodeRemoved', nodeId);
  }

  removeConnection(sourceNodeId, targetNodeId) {
    this.connections = this.connections.filter(
      (conn) => conn.source !== sourceNodeId || conn.target !== targetNodeId
    );
    this.emit('connectionRemoved', sourceNodeId, targetNodeId);
  }

  getNode(nodeId) {
    return this.nodes.get(nodeId);
  }

  getNodes() {
    return Array.from(this.nodes.values());
  }

  getConnections() {
    return this.connections;
  }

  start() {
    this.nodes.forEach(node => {
      if (node.start) {
        node.start();
      }
    });
  }

  stop() {
    this.nodes.forEach(node => {
      if (node.stop) {
        node.stop();
      }
    });
  }
} 