import Node from './Node.js';

class OutputNode extends Node {
    constructor(id, type, config = {}) {
        super(id, type, config);
        this.model = config.model || 'default-model';
        this.prompt = config.prompt || '';
    }

    execute() {
        // Implement the execution logic using this.model and this.prompt
        console.log(`Output Node ${this.id} executed with model ${this.model} and prompt: ${this.prompt}`);
    }
}

export default OutputNode; 