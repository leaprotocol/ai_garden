// @flow
import EventEmitter from 'events';

export class BaseNode extends EventEmitter {
  id: string;
  type: string;
  cursorState: {
    position: number,
    buffer: Array<any>
  };
  
  constructor(id: string, type: string) {
    super();
    this.id = id;
    this.type = type;
    this.cursorState = {
      position: 0,
      buffer: []
    };
  }

  async process(input: any): Promise<any> {
    throw new Error('process() must be implemented by child class');
  }

  async run() {
    const input = await this.getNextInput();
    if (!input) return;
    
    const output = await this.process(input);
    this.cursorState.position++;
    this.emit('processed', output);
  }

  async getNextInput(): Promise<?any> {
    return this.cursorState.buffer[this.cursorState.position];
  }

  reset() {
    this.cursorState = {
      position: 0,
      buffer: []
    };
  }
} 