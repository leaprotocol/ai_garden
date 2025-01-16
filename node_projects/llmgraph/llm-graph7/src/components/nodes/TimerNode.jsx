import React, { useState, useCallback } from 'react';
import { Handle, Position } from 'reactflow';

const TimerNode = ({ id, data }) => {
  const [interval, setInterval] = useState(1000);
  const [isRunning, setIsRunning] = useState(false);
  const [debug, setDebug] = useState(false);
  const [debugOutput, setDebugOutput] = useState('');
  const socket = data.socket;

  const handleIntervalChange = useCallback((event) => {
    setInterval(parseInt(event.target.value, 10) || 1000);
  }, []);

  const handleStartStop = useCallback(() => {
    if (!socket) {
      console.error('No WebSocket connection available');
      if (debug) {
        setDebugOutput(prev => prev + '\n[error] No WebSocket connection');
      }
      return;
    }

    const newIsRunning = !isRunning;
    const message = newIsRunning
      ? { type: 'startTimer', nodeId: id, interval }
      : { type: 'stopTimer', nodeId: id };
    
    if (debug) {
      setDebugOutput(prev => prev + `\n[${newIsRunning ? 'start' : 'stop'}] Sending ${message.type}`);
    }
    
    console.log('Sending message to server:', message);
    socket.send(JSON.stringify(message));
  }, [isRunning, id, interval, socket, debug]);

  // Listen for timer events
  React.useEffect(() => {
    if (!socket) return;

    const handleMessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.nodeId === id) {
        switch (data.type) {
          case 'timerTick':
            if (debug) {
              setDebugOutput(prev => prev + '|');
            }
            break;
            
          case 'timerStarted':
            setIsRunning(true);
            if (debug) {
              setDebugOutput(prev => prev + '\n[confirm] Timer started');
            }
            break;
            
          case 'timerStopped':
            setIsRunning(false);
            if (debug) {
              setDebugOutput(prev => prev + '\n[confirm] Timer stopped');
            }
            break;
            
          case 'nodeError':
            setIsRunning(false);
            if (debug) {
              setDebugOutput(prev => prev + `\n[error] ${data.error}`);
            }
            break;
        }
      }
    };

    socket.addEventListener('message', handleMessage);
    return () => socket.removeEventListener('message', handleMessage);
  }, [socket, id, debug]);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      if (socket && isRunning) {
        socket.send(JSON.stringify({
          type: 'stopTimer',
          nodeId: id
        }));
      }
    };
  }, [socket, id, isRunning]);

  return (
    <div className="timer-node" style={{
      background: '#fff',
      padding: '15px',
      borderRadius: '8px',
      border: '1px solid #ddd',
      minWidth: '200px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      <Handle type="source" position={Position.Right} id="a" />
      
      <div className="node-header" style={{ marginBottom: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <strong>Timer Node {id}</strong>
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
        <label style={{ display: 'block', marginBottom: '5px' }}>Interval (ms):</label>
        <input 
          type="number" 
          value={interval} 
          onChange={handleIntervalChange}
          min="100"
          step="100"
          style={{
            width: '100%',
            padding: '5px',
            borderRadius: '4px',
            border: '1px solid #ddd'
          }}
        />
      </div>

      <button
        onClick={handleStartStop}
        disabled={!socket}
        style={{
          width: '100%',
          padding: '8px',
          background: !socket ? '#ccc' : (isRunning ? '#f44336' : '#4CAF50'),
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: !socket ? 'not-allowed' : 'pointer',
          marginBottom: debug ? '10px' : '0'
        }}
      >
        {!socket ? 'No Connection' : (isRunning ? 'Stop' : 'Start')}
      </button>

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
          <div>Status: {!socket ? 'Disconnected' : (isRunning ? 'Running' : 'Stopped')}</div>
          <div>Interval: {interval}ms</div>
          <div style={{ borderTop: '1px solid #ddd', marginTop: '5px', paddingTop: '5px' }}>
            Debug Output:
            {debugOutput}
          </div>
        </div>
      )}
    </div>
  );
};

export default TimerNode; 