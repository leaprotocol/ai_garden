import React from 'react';

const Sidebar = () => {
  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <aside style={{
      padding: '15px',
      borderRight: '1px solid #eee',
      fontSize: '12px',
      background: '#fcfcfc'
    }}>
      <div className="description" style={{ marginBottom: '10px' }}>
        Drag nodes to the canvas.
      </div>
      
      <div className="dndnode" 
        onDragStart={(event) => onDragStart(event, 'timer')} 
        draggable
        style={{
          padding: '10px',
          border: '1px solid #1a192b',
          borderRadius: '3px',
          marginBottom: '10px',
          cursor: 'grab'
        }}
      >
        Timer Node
      </div>

      <div className="dndnode" 
        onDragStart={(event) => onDragStart(event, 'llm')} 
        draggable
        style={{
          padding: '10px',
          border: '1px solid #1a192b',
          borderRadius: '3px',
          marginBottom: '10px',
          cursor: 'grab'
        }}
      >
        LLM Node
      </div>

      <div className="dndnode" 
        onDragStart={(event) => onDragStart(event, 'output')} 
        draggable
        style={{
          padding: '10px',
          border: '1px solid #1a192b',
          borderRadius: '3px',
          marginBottom: '10px',
          cursor: 'grab'
        }}
      >
        Output Node
      </div>

      <div className="dndnode" 
        onDragStart={(event) => onDragStart(event, 'spawner')} 
        draggable
        style={{
          padding: '10px',
          border: '1px solid #1a192b',
          borderRadius: '3px',
          marginBottom: '10px',
          cursor: 'grab'
        }}
      >
        Analysis Spawner
      </div>

      <div className="dndnode" 
        onDragStart={(event) => onDragStart(event, 'leakyBucket')} 
        draggable
        style={{
          padding: '10px',
          border: '1px solid #1a192b',
          borderRadius: '3px',
          marginBottom: '10px',
          cursor: 'grab',
          background: '#f8f8f8'
        }}
      >
        Leaky Bucket
      </div>

      <div className="dndnode" 
        onDragStart={(event) => onDragStart(event, 'merger')} 
        draggable
        style={{
          padding: '10px',
          border: '1px solid #1a192b',
          borderRadius: '3px',
          marginBottom: '10px',
          cursor: 'grab',
          background: '#f8f8f8'
        }}
      >
        Content Merger
      </div>
    </aside>
  );
};

export default Sidebar; 