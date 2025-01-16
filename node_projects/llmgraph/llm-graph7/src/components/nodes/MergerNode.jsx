import React, { useState, useCallback, useEffect } from 'react';
import { Handle, Position } from 'reactflow';

const MergerNode = ({ id, data }) => {
  const [model, setModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [debug, setDebug] = useState(false);
  const [debugOutput, setDebugOutput] = useState('');
  const [buffer, setBuffer] = useState([]);
  const [mergedResult, setMergedResult] = useState('');
  const socket = data.socket;

  // Listen for available models and responses
  useEffect(() => {
    if (!socket) return;

    const handleMessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'availableModels') {
        setAvailableModels(data.models);
        // Set default model to first available
        if (data.models.length > 0 && !model) {
          setModel(data.models[0].name);
        }
      } else if (data.nodeId === id) {
        if (data.type === 'llmResponseChunk') {
          setMergedResult(prev => prev + data.chunk);
          if (debug) {
            setDebugOutput(prev => prev + '|');
          }
          // Forward merged result
          socket.send(JSON.stringify({
            type: 'nodeOutput',
            nodeId: id,
            output: data.chunk,
            metadata: { source: 'merger' },
            timestamp: data.timestamp,
            formattedTime: data.formattedTime
          }));
        } else if (data.type === 'llmResponseComplete') {
          setIsProcessing(false);
          if (debug) {
            setDebugOutput(prev => prev + '\n[complete]');
          }
          setBuffer([]); // Clear buffer after successful merge
        } else if (data.type === 'llmError') {
          setIsProcessing(false);
          setMergedResult(`Error: ${data.error}`);
          if (debug) {
            setDebugOutput(prev => prev + '\n[error]');
          }
        }
      } else if (data.type === 'nodeOutput' && data.metadata?.source === 'leakyBucket') {
        // Receive content from leaky bucket
        if (debug) {
          setDebugOutput(prev => prev + `\n[received] Content from ${data.nodeId}`);
        }
        
        // Add to buffer
        const newBuffer = [...buffer, { id: data.nodeId, content: data.output }];
        setBuffer(newBuffer);

        // Merge if we have enough content
        if (newBuffer.length >= 2 && !isProcessing) {
          mergeContent(newBuffer);
        }
      }
    };

    socket.addEventListener('message', handleMessage);
    return () => socket.removeEventListener('message', handleMessage);
  }, [socket, id, model, debug, buffer, isProcessing]);

  // Request models on mount
  useEffect(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'getModels' }));
    }
  }, [socket]);

  const mergeContent = useCallback((contentBuffer) => {
    if (!model || isProcessing) return;

    setIsProcessing(true);
    setMergedResult('');

    const prompt = `Here are multiple pieces of content that need to be merged into a coherent summary:

${contentBuffer.map((item, index) => `Content ${index + 1}:\n${item.content}\n`).join('\n')}

Please analyze these pieces and create a comprehensive summary that combines the key insights and information from all sources. The summary should be well-structured and coherent.`;

    socket.send(JSON.stringify({
      type: 'llmRequest',
      nodeId: id,
      prompt,
      model
    }));

    if (debug) {
      setDebugOutput(prev => prev + '\n[merge] Merging content from ' + contentBuffer.length + ' sources');
    }
  }, [socket, id, model, isProcessing, debug]);

  const handleModelChange = useCallback((event) => {
    setModel(event.target.value);
  }, []);

  return (
    <div className="merger-node" style={{
      background: '#fff',
      padding: '15px',
      borderRadius: '8px',
      border: '1px solid #ddd',
      minWidth: '300px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      <Handle type="target" position={Position.Left} />
      
      <div className="node-header" style={{ marginBottom: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <strong>Merger Node {id}</strong>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
            {buffer.length} items buffered
          </div>
        </div>
        <button
          onClick={() => {
            setDebug(!debug);
            if (!debug) setDebugOutput('');
          }}
          style={{
            padding: '4px 8px',
            fontSize: '12px',
            background: debug ? '#2196F3' : '#f5f5f5',
            color: debug ? 'white' : 'black',
            border: '1px solid #ddd',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Debug {debug ? 'On' : 'Off'}
        </button>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <label style={{ display: 'block', marginBottom: '5px' }}>Model:</label>
        <select 
          value={model} 
          onChange={handleModelChange}
          style={{
            width: '100%',
            padding: '5px',
            borderRadius: '4px',
            border: '1px solid #ddd'
          }}
        >
          <option value="">Select a model...</option>
          {availableModels.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name} ({Math.round(m.size / (1024 * 1024))}MB)
            </option>
          ))}
        </select>
      </div>

      {mergedResult && (
        <div style={{ 
          marginTop: '10px',
          padding: '10px',
          background: '#f5f5f5',
          borderRadius: '4px',
          fontSize: '14px',
          whiteSpace: 'pre-wrap',
          maxHeight: '200px',
          overflowY: 'auto'
        }}>
          {mergedResult}
        </div>
      )}

      {debug && (
        <div style={{ 
          marginTop: '10px',
          padding: '10px',
          background: '#f0f0f0',
          borderRadius: '4px',
          fontSize: '12px',
          fontFamily: 'monospace',
          whiteSpace: 'pre-wrap',
          maxHeight: '100px',
          overflowY: 'auto'
        }}>
          <div>Status: {isProcessing ? 'Merging' : 'Ready'}</div>
          <div>Model: {model || 'Not selected'}</div>
          <div>Buffer Size: {buffer.length} items</div>
          <div style={{ borderTop: '1px solid #ddd', marginTop: '5px', paddingTop: '5px' }}>
            Debug Output:
            {debugOutput}
          </div>
        </div>
      )}

      <Handle type="source" position={Position.Right} />
    </div>
  );
};

export default MergerNode; 