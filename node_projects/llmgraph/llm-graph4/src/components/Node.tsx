import { h } from 'preact';
import { useState, useRef, useEffect } from 'preact/hooks';
import { Node as NodeType, Position, NodeData } from '../types';

interface NodeProps extends NodeType {
  onPositionChange?: (id: string, position: Position) => void;
}

export const Node = ({ id, type, position, data, onPositionChange }: NodeProps) => {
  const nodeRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [config, setConfig] = useState(data.config || {});
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  const updateConfig = (key: string, value: unknown) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const renderContent = () => {
    switch(type) {
      case 'llm':
        return (
          <div>
            <select 
              className="model-select" 
              value={config.model || 'gpt-3'} 
              onChange={(e) => updateConfig('model', (e.target as HTMLSelectElement).value)}
            >
              <option value="gpt-3">GPT-3</option>
              <option value="gpt-4">GPT-4</option>
            </select>
            <textarea 
              className="prompt" 
              placeholder="Enter prompt here" 
              value={config.prompt || ''} 
              onInput={(e) => updateConfig('prompt', (e.target as HTMLTextAreaElement).value)}
            />
          </div>
        );
      case 'timer':
        return (
          <div>
            <input 
              type="number" 
              className="interval" 
              value={config.interval || 1000} 
              min="100" 
              onInput={(e) => updateConfig('interval', parseInt((e.target as HTMLInputElement).value, 10))} 
            />
            <button onClick={() => data.onRun?.(id)}>Run</button>
            <div className="output" />
          </div>
        );
      case 'memory':
        return (
          <div>
            <select 
              className="memory-type" 
              value={config.memoryType || 'short'} 
              onChange={(e) => updateConfig('memoryType', (e.target as HTMLSelectElement).value)}
            >
              <option value="short">Short-term Memory</option>
              <option value="long">Long-term Memory</option>
            </select>
            <input 
              type="number" 
              className="memory-size" 
              value={config.maxSize || 10} 
              min="1" 
              step="1" 
              onInput={(e) => updateConfig('maxSize', parseInt((e.target as HTMLInputElement).value, 10))} 
            />
            <select 
              className="retrieval-strategy" 
              value={config.retrievalStrategy || 'fifo'} 
              onChange={(e) => updateConfig('retrievalStrategy', (e.target as HTMLSelectElement).value)}
            >
              <option value="fifo">FIFO</option>
              <option value="relevance">Relevance</option>
            </select>
            <div className="memory-entries" />
            <div className="memory-stats">Stored: <span className="entry-count">0</span></div>
          </div>
        );
      default:
        return data.content;
    }
  };

  return (
    <div 
      ref={nodeRef}
      className={`node ${type} ${isDragging ? 'dragging' : ''}`}
      style={{ 
        transform: `translate(${position.x}px, ${position.y}px)`,
        cursor: isDragging ? 'grabbing' : 'grab'
      }}
    >
      <div className="node-header">
        <span className="node-type">{type.toUpperCase()} Node</span>
        <button onClick={() => data.onDelete?.(id)}>×</button>
      </div>
      <div className="node-content">
        {renderContent()}
      </div>
    </div>
  );
};

export default Node;
