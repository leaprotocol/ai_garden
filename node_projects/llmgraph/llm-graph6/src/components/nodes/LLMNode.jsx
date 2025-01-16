import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';

const LLMNode = memo(({ data }) => {
  return (
    <div className="llm-node">
      <Handle type="target" position={Position.Top} />
      <div className="node-header">LLM Node</div>
      <div className="node-content">
        <select className="model-select" defaultValue={data.model || 'gpt-3.5-turbo'}>
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          <option value="gpt-4">GPT-4</option>
        </select>
        <textarea 
          className="prompt"
          placeholder="Enter prompt..."
          defaultValue={data.prompt || ''}
        />
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

LLMNode.displayName = 'LLMNode';

export default LLMNode; 