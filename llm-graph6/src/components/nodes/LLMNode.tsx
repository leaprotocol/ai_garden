import React, { memo } from 'react';
import Node, { BaseNodeProps } from './Node';

interface LLMNodeProps extends BaseNodeProps {
  data: {
    model?: string;
    prompt?: string;
    selfModification?: boolean;
    onModelChange?: (model: string) => void;
    onPromptChange?: (prompt: string) => void;
    onSelfModificationChange?: (enabled: boolean) => void;
  } & BaseNodeProps['data'];
}

const LLMNode = memo(({ data, ...props }: LLMNodeProps) => {
  return (
    <Node {...props} data={data}>
      <div className="node-content">
        <select 
          className="model-select" 
          value={data.model || 'gpt-3.5-turbo'}
          onChange={(e) => data.onModelChange?.(e.target.value)}
        >
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          <option value="gpt-4">GPT-4</option>
        </select>
        <textarea 
          className="prompt"
          placeholder="Enter prompt..."
          value={data.prompt || ''}
          onChange={(e) => data.onPromptChange?.(e.target.value)}
        />
        <div className="self-modification">
          <label>
            <input
              type="checkbox"
              checked={data.selfModification}
              onChange={(e) => data.onSelfModificationChange?.(e.target.checked)}
            />
            Enable Self-Modification
          </label>
        </div>
      </div>
    </Node>
  );
});

LLMNode.displayName = 'LLMNode';

export default LLMNode; 