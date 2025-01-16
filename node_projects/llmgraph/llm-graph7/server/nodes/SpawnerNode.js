import AbstractNode from './AbstractNode.js';

export default class SpawnerNode extends AbstractNode {
  constructor(id, position) {
    super(id, position);
    this.type = 'spawner';
    this.status = 'ready';
    this.spawnCount = 0;
    this.spawnConfig = {
      nodeType: 'output', // default node type to spawn
      interval: 0, // 0 means manual spawn, >0 means auto-spawn every X ms
      maxNodes: 10, // maximum number of nodes to spawn
      autoConnect: true // whether to automatically connect to spawned nodes
    };
    this.spawnedNodes = new Set(); // keep track of spawned node IDs
    this.intervalId = null;
  }

  handleEvent(event) {
    switch (event.type) {
      case 'updateConfig':
        this.updateSpawnConfig(event.config);
        break;
      case 'spawn':
        this.spawnNode();
        break;
      case 'startAutoSpawn':
        this.startAutoSpawn();
        break;
      case 'stopAutoSpawn':
        this.stopAutoSpawn();
        break;
      case 'reset':
        this.reset();
        break;
      default:
        console.warn(`SpawnerNode: Unknown event type: ${event.type}`);
    }
  }

  updateSpawnConfig(config) {
    this.spawnConfig = { ...this.spawnConfig, ...config };
    this.updateStatus('configured');
    
    // If auto-spawn is running, restart it with new interval
    if (this.intervalId !== null) {
      this.stopAutoSpawn();
      this.startAutoSpawn();
    }
  }

  spawnNode() {
    if (this.spawnedNodes.size >= this.spawnConfig.maxNodes) {
      console.log('SpawnerNode: Maximum number of nodes reached');
      return;
    }

    this.spawnCount++;
    const nodeId = `${this.id}_spawn_${this.spawnCount}`;
    
    // Broadcast spawn request
    this.broadcast({
      type: 'spawnNode',
      nodeType: this.spawnConfig.nodeType,
      nodeId: nodeId,
      position: {
        x: this.position.x + 200, // Offset from spawner
        y: this.position.y + (this.spawnCount * 100) // Stack vertically
      }
    });

    this.spawnedNodes.add(nodeId);

    // Auto-connect if enabled
    if (this.spawnConfig.autoConnect) {
      this.addConnection(nodeId);
    }

    this.updateStatus('spawned');
    
    // Log the spawn event
    this.broadcast({
      type: 'nodeLogUpdate',
      nodeId: this.id,
      log: `Spawned node ${nodeId} (${this.spawnConfig.nodeType})`
    });
  }

  startAutoSpawn() {
    if (this.spawnConfig.interval > 0 && this.intervalId === null) {
      this.intervalId = setInterval(() => {
        this.spawnNode();
      }, this.spawnConfig.interval);
      this.updateStatus('auto-spawning');
    }
  }

  stopAutoSpawn() {
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
      this.updateStatus('ready');
    }
  }

  reset() {
    this.stopAutoSpawn();
    this.spawnCount = 0;
    this.spawnedNodes.clear();
    this.updateStatus('ready');
  }
} 