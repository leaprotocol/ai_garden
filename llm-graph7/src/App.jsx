import React, { useState, useCallback, useRef, useEffect } from 'react';
import ReactFlow, { 
  Background, 
  Controls, 
  ReactFlowProvider,
  useReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  getIncomers,
  getOutgoers,
  getConnectedEdges
} from 'reactflow';
import 'reactflow/dist/style.css';
import BasicNode from './components/nodes/BasicNode';
import TimerNode from './components/nodes/TimerNode';
import OutputNode from './components/nodes/OutputNode';
import LLMNode from './components/nodes/LLMNode';
import SpawnerNode from './components/nodes/SpawnerNode';
import Sidebar from './components/Sidebar';
import LeakyBucketNode from './components/nodes/LeakyBucketNode';
import MergerNode from './components/nodes/MergerNode';

const nodeTypes = {
  basicNode: BasicNode,
  timer: TimerNode,
  output: OutputNode,
  llm: LLMNode,
  spawner: SpawnerNode,
  leakyBucket: LeakyBucketNode,
  merger: MergerNode
};

let id = 0;
const getId = () => `dndnode_${id++}`;

function Flow() {
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [socket, setSocket] = useState(null);

  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  const onConnect = useCallback(
    (params) => {
      console.log('New connection:', params);
      setEdges((eds) => addEdge(params, eds));
      
      if (socket) {
        socket.send(JSON.stringify({
          type: 'connection',
          source: params.source,
          target: params.target
        }));
      }
    },
    [socket]
  );

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow');

      if (typeof type === 'undefined' || !type) {
        return;
      }

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode = {
        id: getId(),
        type,
        position,
        data: { 
          label: `${type} node`,
          socket: socket
        },
        draggable: true
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, socket]
  );

  // Handle spawned nodes and their connections
  useEffect(() => {
    if (!socket) return;

    const handleMessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'nodeSpawned') {
        // Add spawned node to the canvas
        const parentNode = nodes.find(n => n.id === message.parentNodeId);
        if (!parentNode) return;

        const spawnedNode = {
          ...message.nodeData,
          position: {
            x: parentNode.position.x + message.nodeData.position.x,
            y: parentNode.position.y + message.nodeData.position.y
          },
          data: {
            ...message.nodeData.data,
            socket: socket
          }
        };

        setNodes(nds => [...nds, spawnedNode]);

        // Create connection from parent to spawned node
        const newEdge = {
          id: `${message.parentNodeId}-${message.spawnedNodeId}`,
          source: message.parentNodeId,
          target: message.spawnedNodeId,
          type: 'default'
        };

        // Find all outgoing connections from the parent node
        const parentOutgoingEdges = edges.filter(edge => edge.source === message.parentNodeId);
        console.log('Parent outgoing edges:', parentOutgoingEdges);

        // Create new connections from spawned node to parent's targets
        const spawnedOutgoingEdges = parentOutgoingEdges.map(parentEdge => {
          console.log(`Creating connection from spawned node ${message.spawnedNodeId} to target ${parentEdge.target}`);
          return {
            id: `${message.spawnedNodeId}-${parentEdge.target}`,
            source: message.spawnedNodeId,
            target: parentEdge.target,
            type: 'default'
          };
        });
        console.log('Created spawned node connections:', spawnedOutgoingEdges);

        // Add all new edges
        setEdges(eds => {
          const newEdges = [...eds, newEdge, ...spawnedOutgoingEdges];
          console.log('Updated edges:', newEdges);
          return newEdges;
        });

      } else if (message.type === 'nodeRemoved') {
        // Remove the node and its connections
        setNodes(nds => nds.filter(node => node.id !== message.nodeId));
        setEdges(eds => eds.filter(edge => 
          edge.source !== message.nodeId && edge.target !== message.nodeId
        ));
      }
    };

    socket.addEventListener('message', handleMessage);
    return () => socket.removeEventListener('message', handleMessage);
  }, [socket, nodes]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:3000/ws');
    ws.onopen = () => {
      console.log('Connected to WebSocket');
      setSocket(ws);
    };
    ws.onclose = () => console.log('WebSocket disconnected');
    ws.onerror = (error) => console.error('WebSocket error:', error);

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div className="dndflow">
      <Sidebar />
      <div ref={reactFlowWrapper} className="reactflow-wrapper" style={{ width: '100%', height: '100vh' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onInit={setReactFlowInstance}
          onDrop={onDrop}
          onDragOver={onDragOver}
          fitView
          panOnDrag={[1, 2]}
        >
          <Background />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <Flow />
    </ReactFlowProvider>
  );
} 