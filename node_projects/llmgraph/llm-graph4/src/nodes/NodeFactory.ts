import { NodeType, NodeConfig } from '../types';

interface NodeInstance {
  id: string;
  type: NodeType;
  config: NodeConfig;
  run: () => Promise<void>;
  process?: (input: unknown) => Promise<unknown>;
  on: (event: string, callback: (data: unknown) => void) => void;
}

class BaseNode implements NodeInstance {
  id: string;
  type: NodeType;
  config: NodeConfig;
  private eventHandlers: Record<string, ((data: unknown) => void)[]> = {};

  constructor(id: string, type: NodeType, config: NodeConfig) {
    this.id = id;
    this.type = type;
    this.config = config;
  }

  async run() {
    // Base implementation
  }

  on(event: string, callback: (data: unknown) => void) {
    if (!this.eventHandlers[event]) {
      this.eventHandlers[event] = [];
    }
    this.eventHandlers[event].push(callback);
  }

  protected emit(event: string, data: unknown) {
    this.eventHandlers[event]?.forEach(handler => handler(data));
  }
}

class LLMNode extends BaseNode {
  async process(input: unknown): Promise<unknown> {
    const { model, prompt } = this.config;
    // Implement LLM processing logic here
    const result = `Processed by ${model}: ${input}`;
    this.emit('processed', result);
    return result;
  }
}

class TimerNode extends BaseNode {
  private intervalId?: number;

  async run() {
    if (this.intervalId) {
      window.clearInterval(this.intervalId);
    }
    
    this.intervalId = window.setInterval(() => {
      this.emit('processed', Date.now());
    }, this.config.interval || 1000);
  }

  stop() {
    if (this.intervalId) {
      window.clearInterval(this.intervalId);
    }
  }
}

class MemoryNode extends BaseNode {
  private memory: unknown[] = [];

  async process(input: unknown): Promise<unknown> {
    const { maxSize = 10, retrievalStrategy = 'fifo' } = this.config;
    
    if (this.memory.length >= maxSize) {
      if (retrievalStrategy === 'fifo') {
        this.memory.shift();
      }
    }
    
    this.memory.push(input);
    this.emit('processed', this.memory);
    return this.memory;
  }

  clear() {
    this.memory = [];
    this.emit('cleared', null);
  }
}

export class NodeFactory {
  static createNode(type: NodeType, id: string, config: NodeConfig): NodeInstance {
    switch (type) {
      case 'llm':
        return new LLMNode(id, type, config);
      case 'timer':
        return new TimerNode(id, type, config);
      case 'memory':
        return new MemoryNode(id, type, config);
      default:
        throw new Error(`Unknown node type: ${type}`);
    }
  }
}
