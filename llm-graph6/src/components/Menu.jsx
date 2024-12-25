import React from 'react';

function Menu() {
  const handleAddNode = (type) => {
    const position = { x: Math.random() * 500, y: Math.random() * 300 };
    window.dispatchEvent(new CustomEvent('add-node', { detail: { type, position } }));
  };

  return (
    <div style={{ padding: '10px', borderBottom: '1px solid #ccc' }}>
      <button onClick={() => handleAddNode('timer')}>Add Timer</button>
      <button onClick={() => handleAddNode('output')}>Add Output</button>
      <button>Clear</button>
      <button>Save</button>
      <button>Load</button>
    </div>
  );
}

export default Menu; 