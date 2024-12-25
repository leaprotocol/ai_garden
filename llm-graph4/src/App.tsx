import { h } from 'preact';
import { useState, useCallback } from 'preact/hooks';
import ReactFlow, { 
  Controls, 
  Background,
  addEdge,
  Connection,
  Edge,
  Node as FlowNode
} from 'reactflow';
import 'reactflow/dist/style.css';

import { NodeFactory } from './nodes/NodeFactory';
import { CustomNode } from './components/CustomNode';
import Menu from './components/Menu';
import { NodeType } from './types';

const nodeTypes = {
  llm: CustomNode,
  timer: CustomNode,
  memory: CustomNode,
  merger: CustomNode,
  output: CustomNode,
};

export function App() {
  const [nodes, setNodes] = useState<FlowNode[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [nodeInstances, setNodeInstances] = useState<Record<string, any>>({});

  const onConnect = useCallback((params: Connection) => {
    setEdges(prev => addEdge(params, prev));
  }, []);

  const addNode = (type: NodeType) => {
    const id = crypto.randomUUID();
    const instance = NodeFactory.createNode(type, id, {});
    
    setNodeInstances(prev => ({
      ...prev,
      [id]: instance
    }));

    const newNode: FlowNode = {
      id,
      type,
      position: { x: 100, y: 100 },
      data: { 
        label: `${type.toUpperCase()} Node`,
        onRun: () => runNode(id),
        onDelete: () => deleteNode(id)
      }
    };

    setNodes(prev => [...prev, newNode]);
  };

  const runNode = async (id: string) => {
    const instance = nodeInstances[id];
    if (!instance) return;

    const outEdges = edges.filter(e => e.source === id);
    instance.on('processed', async (data) => {
      for (const edge of outEdges) {
        const targetInstance = nodeInstances[edge.target];
        if (targetInstance?.process) {
          await targetInstance.process(data);
        }
      }
    });

    await instance.run();
  };

  const deleteNode = (id: string) => {
    setNodes(prev => prev.filter(node => node.id !== id));
    setEdges(prev => prev.filter(edge => 
      edge.source !== id && edge.target !== id
    ));
    setNodeInstances(prev => {
      const { [id]: removed, ...rest } = prev;
      return rest;
    });
  };

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Menu addNode={addNode} />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={setNodes}
        onEdgesChange={setEdges}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}