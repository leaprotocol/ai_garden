import React, { useState, useCallback } from 'react';
import { Handle, Position } from 'reactflow';

const SpawnerNode = ({ id, data }) => {
  const [nodeType, setNodeType] = useState('llm');
  const [maxNodes, setMaxNodes] = useState(3);
  const [autoCleanupTimeout, setAutoCleanupTimeout] = useState(60000);
  const [debug, setDebug] = useState(false);
  const [debugOutput, setDebugOutput] = useState('');
  const [spawnedNodes, setSpawnedNodes] = useState([]);
  const [topic, setTopic] = useState('artificial intelligence');
  const socket = data.socket;

  // Predefined analysis prompts
  const prompts = [
    {
      role: "Technical Analysis",
      prompt: `Analyze the technical aspects of ${topic}. Focus on current capabilities and limitations. Be specific and concise.`
    },
    {
      role: "Social Impact",
      prompt: `Analyze the social impact of ${topic} on society and daily life. Consider both positive and negative effects. Be specific and concise.`
    },
    {
      role: "Future Trends",
      prompt: `Predict the future trends and developments in ${topic} over the next 5 years. Focus on major changes and innovations. Be specific and concise.`
    }
  ];

  const handleNodeTypeChange = useCallback((event) => {
    setNodeType(event.target.value);
  }, []);

  const handleMaxNodesChange = useCallback((event) => {
    setMaxNodes(parseInt(event.target.value, 10) || 1);
  }, []);

  const handleTimeoutChange = useCallback((event) => {
    setAutoCleanupTimeout(parseInt(event.target.value, 10) || 60000);
  }, []);

  const handleTopicChange = useCallback((event) => {
    setTopic(event.target.value);
  }, []);

  const handleSpawnAll = useCallback(() => {
    if (socket) {
      prompts.forEach((promptData, index) => {
        if (spawnedNodes.length < maxNodes) {
          const spawnRequest = {
            type: 'spawnNode',
            nodeId: id,
            nodeType: nodeType,
            timeout: autoCleanupTimeout,
            promptData: {
              role: promptData.role,
              prompt: promptData.prompt.replace('${topic}', topic)
            },
            timestamp: Date.now(),
            formattedTime: new Date().toLocaleTimeString()
          };

          socket.send(JSON.stringify(spawnRequest));
          if (debug) {
            setDebugOutput(prev => prev + `\n[spawn] Requesting new node for ${promptData.role}`);
          }
        }
      });
    }
  }, [id, nodeType, autoCleanupTimeout, maxNodes, spawnedNodes.length, socket, debug, topic]);

  // Listen for spawned node events
  React.useEffect(() => {
    if (!socket) return;

    const handleMessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.parentNodeId === id) {
        if (data.type === 'nodeSpawned') {
          setSpawnedNodes(prev => [...prev, data.spawnedNodeId]);
          if (debug) {
            setDebugOutput(prev => prev + `\n[created] Node ${data.spawnedNodeId} for ${data.promptData?.role || 'unknown role'}`);
          }
        } else if (data.type === 'nodeRemoved') {
          setSpawnedNodes(prev => prev.filter(nodeId => nodeId !== data.nodeId));
          if (debug) {
            setDebugOutput(prev => prev + `\n[completed] Node ${data.nodeId}`);
          }
        } else if (data.type === 'nodeError') {
          if (debug) {
            setDebugOutput(prev => prev + `\n[error] Node ${data.nodeId}: ${data.error}`);
          }
        }
      }
    };

    socket.addEventListener('message', handleMessage);
    return () => socket.removeEventListener('message', handleMessage);
  }, [socket, id, debug]);

  return (
    <div className="spawner-node" style={{
      background: '#fff',
      padding: '15px',
      borderRadius: '8px',
      border: '1px solid #ddd',
      minWidth: '300px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      <Handle type="target" position={Position.Left} />
      
      <div className="node-header" style={{ marginBottom: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <strong>Analysis Spawner {id}</strong>
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

      <div className="node-content">
        <div style={{ marginBottom: '10px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>Analysis Topic:</label>
          <input
            type="text"
            value={topic}
            onChange={handleTopicChange}
            placeholder="Enter topic to analyze..."
            style={{
              width: '100%',
              padding: '5px',
              borderRadius: '4px',
              border: '1px solid #ddd'
            }}
          />
        </div>

        <div style={{ marginBottom: '10px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>Max Concurrent Analyses:</label>
          <input
            type="number"
            value={maxNodes}
            onChange={handleMaxNodesChange}
            min="1"
            max="10"
            style={{
              width: '100%',
              padding: '5px',
              borderRadius: '4px',
              border: '1px solid #ddd'
            }}
          />
        </div>

        <div style={{ marginBottom: '10px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>Auto-cleanup Timeout (ms):</label>
          <input
            type="number"
            value={autoCleanupTimeout}
            onChange={handleTimeoutChange}
            min="1000"
            step="1000"
            style={{
              width: '100%',
              padding: '5px',
              borderRadius: '4px',
              border: '1px solid #ddd'
            }}
          />
        </div>

        <button
          onClick={handleSpawnAll}
          disabled={spawnedNodes.length >= maxNodes}
          style={{
            width: '100%',
            padding: '8px',
            background: spawnedNodes.length >= maxNodes ? '#ccc' : '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: spawnedNodes.length >= maxNodes ? 'not-allowed' : 'pointer',
            marginBottom: '10px'
          }}
        >
          Spawn Analysis Nodes ({spawnedNodes.length}/{maxNodes})
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
            maxHeight: '150px',
            overflowY: 'auto'
          }}>
            <div>Analysis Roles:</div>
            {prompts.map((p, i) => (
              <div key={i} style={{ marginLeft: '10px', color: '#666' }}>• {p.role}</div>
            ))}
            <div style={{ marginTop: '10px', borderTop: '1px solid #ddd', paddingTop: '10px' }}>
              Debug Output:
              {debugOutput}
            </div>
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  );
};

export default SpawnerNode; 