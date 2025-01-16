// @flow
import { BaseNode } from './BaseNode';
import type { ProcessConfig } from '../types';

export class SinkNode extends BaseNode {
  process: (input: any) => Promise<void>;
  
  constructor(id: string, config: ProcessConfig) {
    super(id, 'sink');
    this.process = config.process;
  }
} 