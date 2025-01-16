import { h } from 'preact';
import { Handle, Position, NodeProps } from 'reactflow';

export function CustomNode({ data, type }: NodeProps) {
  return (
    <div className="custom-node">
      <Handle type="target" position={Position.Top} />
      <div className="node-header">
        {data.label}
        <button onClick={data.onDelete}>×</button>
      </div>
      <div className="node-content">
        {type === 'llm' && (
          <select className="model-select">
            <option value="gpt-3">GPT-3</option>
            <option value="gpt-4">GPT-4</option>
          </select>
        )}
        <button onClick={data.onRun}>Run</button>
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
} 