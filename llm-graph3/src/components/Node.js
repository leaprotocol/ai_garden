// @flow
import { h } from 'preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import type { NodeProps } from '../types';

export function Node({ id, type, data, position, onPositionChange }: NodeProps) {
  const nodeRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  const handleMouseDown = (e) => {
    setIsDragging(true);
    const rect = nodeRef.current?.getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    
    const x = e.clientX - dragOffset.x;
    const y = e.clientY - dragOffset.y;
    onPositionChange(id, { x, y });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  return (
    <div 
      ref={nodeRef}
      class={`node ${type} ${isDragging ? 'dragging' : ''}`}
      style={{ 
        transform: `translate(${position.x}px, ${position.y}px)`,
        cursor: isDragging ? 'grabbing' : 'grab'
      }}
      onMouseDown={handleMouseDown}
    >
      <div class="node-header">
        <span class="node-type">{type}</span>
        <div class="node-controls">
          <button onClick={() => data.onRun?.(id)}>Run</button>
          <button onClick={() => data.onDelete?.(id)}>×</button>
        </div>
      </div>
      <div class="node-content">
        {data.content}
      </div>
      <div class="node-ports">
        <div class="input-ports">
          {data.inputs?.map(port => (
            <div class="port input" data-port-id={port.id} />
          ))}
        </div>
        <div class="output-ports">
          {data.outputs?.map(port => (
            <div class="port output" data-port-id={port.id} />
          ))}
        </div>
      </div>
    </div>
  );
} 