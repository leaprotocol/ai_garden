// @flow
import { BaseNode } from './BaseNode';
import { SourceNode } from './SourceNode';
import type { NodeConfig } from '../types';

export class NodeFactory {
  static createNode(type: string, config: NodeConfig): BaseNode {
    switch (type) {
      case 'source':
        return new SourceNode(crypto.randomUUID(), config);
      case 'processor':
        return new ProcessorNode(crypto.randomUUID(), config);
      case 'sink':
        return new SinkNode(crypto.randomUUID(), config);
      default:
        throw new Error(`Unknown node type: ${type}`);
    }
  }
} 