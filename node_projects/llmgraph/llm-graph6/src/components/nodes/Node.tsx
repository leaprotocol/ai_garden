import React, { memo } from 'react';
import { Handle, Position, NodeProps as FlowNodeProps } from 'reactflow';

export interface BaseNodeProps extends FlowNodeProps {
  data: {
    label: string;
    onRun?: () => void;
    onDelete?: () => void;
    config?: Record<string, any>;
  };
}

const Node = memo(({ data, id, type, xPos, yPos }: BaseNodeProps) => {
  const handleRunClick = () => {
    if (data.onRun) {
      data.onRun();
    }
  };

  const handleDeleteClick = () => {
    if (data.onDelete) {
      data.onDelete();
    }
  };

  return (
    <div
      className={`${type}-node`}
      style={{ position: 'absolute', left: xPos, top: yPos }}
      draggable
    >
      <Handle type="target" position={Position.Top} />
      <div className="node-header">
        {data.label}
        <div className="node-controls">
          <button onClick={handleRunClick} className="run-button">▶</button>
          <button onClick={handleDeleteClick} className="delete-button">×</button>
        </div>
      </div>
      <div className="node-content">
        {/* Child components will render their specific content here */}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

Node.displayName = 'Node';

export default Node; 