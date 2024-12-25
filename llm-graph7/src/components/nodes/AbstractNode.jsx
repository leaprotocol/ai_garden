import React from 'react';
import { Handle, Position } from 'reactflow';
import { logger } from '../../utils/logger.js';

const AbstractNode = ({ id, data, children, hasInput = true, hasOutput = true }) => {
  const emitEvent = (eventData) => {
    logger.info(`Node ${id}: Emitting event:`, eventData);
    if (data.onNodeDataChange) {
      data.onNodeDataChange(id, {
        lastEvent: {
          ...eventData,
          source: id,
          timestamp: Date.now(),
          formattedTime: new Date().toLocaleTimeString()
        }
      });
    } else {
      logger.warn(`Node ${id}: onNodeDataChange not provided`);
    }
  };

  return (
    <div className={`node ${data.label.toLowerCase()}-node`}>
      {hasInput && <Handle type="target" position={Position.Left} />}
      
      <div className="node-header">
        {data.label}
      </div>
      
      <div className="node-content">
        {React.Children.map(children, child => {
          if (React.isValidElement(child)) {
            return React.cloneElement(child, { emitEvent });
          }
          return child;
        })}
      </div>

      {hasOutput && <Handle type="source" position={Position.Right} />}
    </div>
  );
};

export default AbstractNode; 