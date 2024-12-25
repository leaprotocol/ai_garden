// @flow
import { BaseNode } from './BaseNode';
import { createReadStream } from 'fs';
import { stat } from 'fs/promises';

export class SourceNode extends BaseNode {
  path: string;
  chunkSize: number;
  
  constructor(id: string, config: { path: string, chunkSize: number }) {
    super(id, 'source');
    this.path = config.path;
    this.chunkSize = config.chunkSize || 1000;
  }

  async process(input: any): Promise<any> {
    const stats = await stat(this.path);
    const stream = createReadStream(this.path, { 
      start: this.cursorState.position * this.chunkSize,
      end: (this.cursorState.position + 1) * this.chunkSize - 1,
      encoding: 'utf8'
    });

    return new Promise((resolve, reject) => {
      let content = '';
      stream.on('data', chunk => content += chunk);
      stream.on('end', () => resolve({
        type: 'chunk',
        content,
        metadata: {
          path: this.path,
          position: this.cursorState.position,
          isEOF: content.length < this.chunkSize
        }
      }));
      stream.on('error', reject);
    });
  }
} 