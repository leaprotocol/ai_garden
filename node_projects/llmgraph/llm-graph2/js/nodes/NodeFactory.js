import Node from './Node.js';
import LLMNode from './LLMNode.js';
import TimerNode from './TimerNode.js';
import MemoryNode from './MemoryNode.js';
import MergerNode from './MergerNode.js';
import OutputNode from './OutputNode.js';

class NodeFactory {
    static createNode(id, type, config = {}) {
        switch(type) {
            case 'llm':
                return new LLMNode(id, type, config);
            case 'timer':
                return new TimerNode(id, type, config);
            case 'memory':
                return new MemoryNode(id, type, config);
            case 'merger':
                return new MergerNode(id, type, config);
            case 'output':
                return new OutputNode(id, type, config);
            default:
                return new Node(id, type, config);
        }
    }
}

export default NodeFactory; 