import React, { useState, useCallback, useEffect } from 'react';
import { Handle } from 'reactflow';
import { createEvent, createChunkEvent, createCompleteEvent, createErrorEvent, EventTypes } from '../../types/events';

export default function LLMNode({ id, data }) {
  const [prompt, setPrompt] = useState(data.promptData?.prompt || '');
  const [model, setModel] = useState('smollm:latest');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDebug, setIsDebug] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [response, setResponse] = useState('');
  const socket = data.socket;

  const debugLog = useCallback((...args) => {
    if (isDebug) {
      console.log(`[LLM ${id}]`, ...args);
    }
  }, [isDebug, id]);

  // Fetch available models on mount
  useEffect(() => {
    if (socket) {
      socket.send(JSON.stringify({ type: 'getModels' }));
    }
  }, [socket]);

  // Handle incoming messages
  useEffect(() => {
    if (!socket) return;

    const handleMessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'availableModels') {
        setAvailableModels(message.models);
      }
      else if (message.nodeId === id) {
        switch (message.type) {
          case 'llmResponseChunk':
            setResponse(prev => prev + message.chunk);
            // Create and emit chunk event
            if (socket) {
              const chunkEvent = createChunkEvent(id, message.chunk);
              socket.send(JSON.stringify(chunkEvent));
            }
            break;
          case 'llmResponseComplete':
            setIsProcessing(false);
            // Create and emit complete event
            if (socket) {
              const completeEvent = createCompleteEvent(id, response);
              socket.send(JSON.stringify(completeEvent));
            }
            break;
          case 'llmError':
            setIsProcessing(false);
            // Create and emit error event
            if (socket) {
              const errorEvent = createErrorEvent(id, message.error);
              socket.send(JSON.stringify(errorEvent));
            }
            break;
        }
      }
    };

    socket.addEventListener('message', handleMessage);
    return () => socket.removeEventListener('message', handleMessage);
  }, [socket, id, response]);

  // Handle prompt submission
  const handleSubmit = useCallback(() => {
    if (!socket || !prompt || isProcessing) return;

    debugLog('Submitting prompt:', prompt);
    setIsProcessing(true);
    setResponse('');

    // Send LLM request
    socket.send(JSON.stringify({
      type: 'llmRequest',
      nodeId: id,
      prompt,
      model
    }));
  }, [socket, id, prompt, model, isProcessing, debugLog]);

  // Handle incoming events
  const handleEvent = useCallback((evt) => {
    if (!socket || isProcessing) return;

    debugLog('Received event:', evt);
    
    // Use the incoming event's data as the prompt
    if (typeof evt.data === 'string') {
      setPrompt(evt.data);
      handleSubmit();
    }
  }, [socket, isProcessing, handleSubmit, debugLog]);

  return (
    <div className="node llm-node">
      <Handle type="target" position="left" />
      
      <div className="node-content">
        <h4>LLM Node</h4>
        <div className="node-controls">
          <select 
            value={model} 
            onChange={e => setModel(e.target.value)}
            disabled={isProcessing}
          >
            {availableModels.map(m => (
              <option key={m.name} value={m.name}>{m.name}</option>
            ))}
          </select>
          
          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Enter prompt..."
            disabled={isProcessing}
          />
          
          <button 
            onClick={handleSubmit}
            disabled={isProcessing || !prompt}
          >
            {isProcessing ? 'Processing...' : 'Submit'}
          </button>
          
          <button onClick={() => setIsDebug(prev => !prev)}>
            {isDebug ? '🐛 Debug On' : '🪲 Debug Off'}
          </button>
        </div>
        
        {response && (
          <div className="node-response">
            <strong>Response:</strong>
            <pre>{response}</pre>
          </div>
        )}
      </div>
      
      <Handle type="source" position="right" />
    </div>
  );
} 