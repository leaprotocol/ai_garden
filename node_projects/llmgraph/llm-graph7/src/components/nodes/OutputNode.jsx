import React, { useState, useCallback, useEffect } from 'react';
import { Handle, Position, useEdges } from 'reactflow';

const OutputNode = ({ id, data }) => {
  const [receivedEvents, setReceivedEvents] = useState([]);
  const [isActive, setIsActive] = useState(true);
  const [debug, setDebug] = useState(false);
  const [debugOutput, setDebugOutput] = useState('');
  const socket = data.socket;
  const edges = useEdges();

  useEffect(() => {
    if (!socket) return;

    const handleMessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Check if there's a connection to the source node
      const isConnected = edges.some(edge => edge.target === id && edge.source === data.nodeId);
      if (!isConnected) return;
      
      if (isActive) {
        if (data.type === 'timerTick') {
          setReceivedEvents(prev => [...prev, {
            type: 'Timer Tick',
            source: data.nodeId,
            timestamp: data.timestamp,
            formattedTime: data.formattedTime
          }]);
          if (debug) setDebugOutput(prev => prev + '|');
        } else if (data.type === 'content') {
          setReceivedEvents(prev => [...prev, {
            type: data.metadata?.source || 'Content',
            source: data.nodeId,
            content: data.content,
            metadata: data.metadata,
            timestamp: data.timestamp,
            formattedTime: data.formattedTime
          }]);
          if (debug) {
            const status = data.metadata?.isComplete ? '[complete]' : '|';
            setDebugOutput(prev => prev + status);
          }
        } else if (data.type === 'llmResponseChunk' || data.type === 'nodeOutput') {
          // Handle legacy event types during transition
          const content = data.type === 'llmResponseChunk' ? data.chunk : data.output;
          setReceivedEvents(prev => [...prev, {
            type: data.type === 'llmResponseChunk' ? 'LLM Chunk' : 'Node Output',
            source: data.nodeId,
            content: content,
            timestamp: data.timestamp,
            formattedTime: data.formattedTime
          }]);
          if (debug) setDebugOutput(prev => prev + '|');
        }
      }
    };

    socket.addEventListener('message', handleMessage);

    return () => {
      socket.removeEventListener('message', handleMessage);
    };
  }, [socket, isActive, id, edges, debug]);

  const handleClear = useCallback(() => {
    setReceivedEvents([]);
    setDebugOutput('');
  }, []);

  const toggleActive = useCallback(() => {
    setIsActive(prev => !prev);
  }, []);

  return (
    <div className="output-node" style={{ background: '#fff', padding: '10px', borderRadius: '5px', border: '1px solid #ddd', minWidth: '200px' }}>
      <Handle type="target" position={Position.Left} id="a" />
      <div className="node-header" style={{ marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <strong>Output Node {id}</strong>
        <button onClick={toggleActive} style={{
          padding: '4px 8px',
          fontSize: '12px',
          background: isActive ? '#2196F3' : '#f5f5f5',
          color: isActive ? 'white' : 'black',
          border: '1px solid #ddd',
          borderRadius: '4px',
          cursor: 'pointer'
        }}>
          {isActive ? 'Deactivate' : 'Activate'}
        </button>
        <button onClick={handleClear} style={{
          padding: '4px 8px',
          fontSize: '12px',
          background: '#f5f5f5',
          border: '1px solid #ddd',
          borderRadius: '4px',
          cursor: 'pointer'
        }}>
          Clear
        </button>
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

      {debug && debugOutput && (
        <div style={{ 
          marginBottom: '10px',
          padding: '5px',
          background: '#f0f0f0',
          borderRadius: '4px',
          fontSize: '12px',
          fontFamily: 'monospace',
          whiteSpace: 'pre-wrap',
          maxHeight: '50px',
          overflowY: 'auto'
        }}>
          Debug: {debugOutput}
        </div>
      )}

      <div className="output-content" style={{ 
        maxHeight: '200px', 
        overflowY: 'auto', 
        padding: '5px',
        background: '#f5f5f5',
        borderRadius: '3px'
      }}>
        {isActive ? (
          receivedEvents.length === 0 ? (
            <div style={{ color: '#666', fontStyle: 'italic' }}>No events received yet</div>
          ) : (
            <div>
              {receivedEvents.map((event, index) => (
                <div key={index} style={{ 
                  marginBottom: '5px', 
                  fontSize: '12px',
                  padding: '3px',
                  borderBottom: '1px solid #ddd'
                }}>
                  {event.formattedTime}: {event.type} from Node {event.source}
                  {event.content && (
                    <div style={{ 
                      marginLeft: '10px',
                      marginTop: '2px',
                      whiteSpace: 'pre-wrap',
                      fontFamily: 'monospace',
                      fontSize: '11px'
                    }}>
                      {event.content}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )
        ) : (
          <div style={{ color: '#666', fontStyle: 'italic' }}>Output is deactivated</div>
        )}
      </div>
    </div>
  );
};

export default OutputNode; 