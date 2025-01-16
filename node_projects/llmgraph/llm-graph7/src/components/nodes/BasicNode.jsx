import React from 'react';
import { Handle, Position } from 'reactflow';

const BasicNode = ({ data }) => {
  return (
    <>
      <Handle type="source" position={Position.Right} />
      <div 
        style={{
          background: '#fff',
          border: '1px solid #222',
          padding: '10px',
          borderRadius: '3px',
          width: '150px',
          cursor: 'grab'
        }}
      >
        <div>Basic Node</div>
        <div>{data.label}</div>
      </div>
      <Handle type="target" position={Position.Left} />
    </>
  );
};

export default BasicNode; 