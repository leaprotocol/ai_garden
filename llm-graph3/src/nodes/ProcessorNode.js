// @flow
import { BaseNode } from './BaseNode';
import type { ProcessConfig } from '../types';

export class ProcessorNode extends BaseNode {
  process: (input: any) => Promise<any>;
  
  constructor(id: string, config: ProcessConfig) {
    super(id, 'processor');
    this.process = config.process;
  }
} 