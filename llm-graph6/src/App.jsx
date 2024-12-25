import React, { useRef, useState, useMemo } from 'react';
import { ReactFlowProvider } from 'reactflow';
import GraphCanvas from './components/GraphCanvas';
import TimerNode from './components/nodes/TimerNode';
import LLMNode from './components/nodes/LLMNode';
import OutputNode from './components/nodes/OutputNode';
// Import other node types as needed

// Define nodeTypes outside the component
const nodeTypes = {
  llm: LLMNode,
  timer: TimerNode,
  output: OutputNode,
  // Add other node types here
};

export default function App() {
  const canvasRef = useRef();
  const [showOutputNode, setShowOutputNode] = useState(true);

  const handleAddTimer = () => {
    canvasRef.current?.onAddNode('timer');
  };

  const handleAddOutput = () => {
    canvasRef.current?.onAddNode('output');
  };

  const handleAddLLM = () => {
    canvasRef.current?.onAddNode('llm');
  };

  const handleClear = () => {
    canvasRef.current?.clearGraph();
  };

  const handleSave = () => {
    canvasRef.current?.saveGraph();
  };

  const handleLoad = () => {
    canvasRef.current?.loadGraph();
  };

  const toggleOutputNode = () => {
    if (showOutputNode) {
      canvasRef.current?.onRemoveNode('output');
    } else {
      canvasRef.current?.onAddNode('output');
    }
    setShowOutputNode(prev => !prev);
  };

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div className="toolbar" style={{ padding: '10px' }}>
        <button onClick={handleAddTimer}>Add Timer</button>
        <button onClick={handleAddOutput}>Add Output</button>
        <button onClick={handleAddLLM}>Add LLM</button>
        <button onClick={handleClear}>Clear</button>
        <button onClick={handleSave}>Save</button>
        <button onClick={handleLoad}>Load</button>
        <button onClick={toggleOutputNode}>
          {showOutputNode ? 'Deactivate OutputNode' : 'Activate OutputNode'}
        </button>
      </div>
      <div style={{ flex: 1, position: 'relative' }}>
        <ReactFlowProvider>
          <GraphCanvas ref={canvasRef} nodeTypes={nodeTypes} />
        </ReactFlowProvider>
      </div>
    </div>
  );
}