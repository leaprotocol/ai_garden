// @flow
import { h } from 'preact';
import { useState } from 'preact/hooks';

type NodeType = 'llm' | 'timer' | 'memory' | 'merger' | 'output';

type NodeConfig = {
  [key: string]: any
};

type Props = {
  node: {
    id: number,
    type: NodeType,
    config: NodeConfig,
    position: {
      top: string,
      left: string
    }
  }
};

const Node = ({ node }: Props) => {
  const [config, setConfig] = useState<NodeConfig>(node.config);

  const updateConfig = (key: string, value: any) => {
    setConfig({ ...config, [key]: value });
  };

  const renderContent = () => {
    switch(node.type) {
      case 'llm':
        return (
          <div>
            <select className="model-select" value={config.model || 'gpt-3'} onChange={(e) => updateConfig('model', e.target.value)}>
              <option value="gpt-3">GPT-3</option>
              <option value="gpt-4">GPT-4</option>
            </select>
            <textarea className="prompt" placeholder="Enter prompt here" value={config.prompt || ''} onInput={(e) => updateConfig('prompt', e.target.value)}></textarea>
          </div>
        );
      case 'timer':
        return (
          <div>
            <input type="number" className="interval" value={config.interval || 1000} min="100" onInput={(e) => updateConfig('interval', parseInt(e.target.value, 10))} />
            <button onClick={() => runNode(node.id)}>Run</button>
            <div className="output"></div>
          </div>
        );
      case 'memory':
        return (
          <div>
            <select className="memory-type" value={config.memoryType || 'short'} onChange={(e) => updateConfig('memoryType', e.target.value)}>
              <option value="short">Short-term Memory</option>
              <option value="long">Long-term Memory</option>
            </select>
            <input type="number" className="memory-size" value={config.maxSize || 10} min="1" step="1" onInput={(e) => updateConfig('maxSize', parseInt(e.target.value, 10))} />
            <label>max entries</label>
            <select className="retrieval-strategy" value={config.retrievalStrategy || 'fifo'} onChange={(e) => updateConfig('retrievalStrategy', e.target.value)}>
              <option value="fifo">FIFO</option>
              <option value="relevance">Relevance</option>
            </select>
            <button onClick={() => clearMemory(node.id)}>Clear Memory</button>
            <div className="memory-entries"></div>
            <div className="memory-stats">Stored: <span className="entry-count">0</span></div>
          </div>
        );
      case 'merger':
        return (
          <div>
            <select className="model-select" value={config.model || 'gpt-3'} onChange={(e) => updateConfig('model', e.target.value)}>
              <option value="gpt-3">GPT-3</option>
              <option value="gpt-4">GPT-4</option>
            </select>
            <textarea className="prompt" placeholder="Enter merger prompt here" value={config.prompt || ''} onInput={(e) => updateConfig('prompt', e.target.value)}></textarea>
          </div>
        );
      case 'output':
      default:
        return (
          <div className="output"></div>
        );
    }
  };

  const runNode = (id: number) => {
    // Implement run logic
    console.log(`Running node ${id}`);
  };

  const clearMemory = (id: number) => {
    // Implement clear memory logic
    console.log(`Clearing memory for node ${id}`);
  };

  return (
    <div className="node" style={{ top: node.position.top, left: node.position.left }}>
      <div className="node-header">
        {`${node.type.toUpperCase()} Node`}
      </div>
      <div className="node-content">
        {renderContent()}
      </div>
    </div>
  );
};

export default Node; 