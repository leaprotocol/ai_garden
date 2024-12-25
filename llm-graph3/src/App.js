// @flow
import { h } from 'preact';
import { useState, useEffect } from 'preact/hooks';
import { NodeFactory } from './nodes/NodeFactory';
import { Node as NodeComponent } from './components/Node';

type Position = {|
  x: number,
  y: number
|};

type Port = {|
  id: string,
  type: string,
  data?: any
|};

type NodeType = 'source' | 'processor' | 'sink';

type NodeData = {|
  content: any,
  inputs?: Array<Port>,
  outputs?: Array<Port>,
  cursorState?: {
    position: number,
    buffer: Array<any>
  }
|};

type Node = {|
  id: string,
  type: NodeType,
  position: Position,
  data: NodeData
|};

type Connection = {|
  id: string,
  source: string,
  target: string
|};

type NodeConfig = {|
  path?: string,
  chunkSize?: number,
  process?: (input: any) => Promise<any>
|};

export function App() {
  const [nodes, setNodes] = useState<Array<Node>>([]);
  const [nodeInstances, setNodeInstances] = useState({});
  const [connections, setConnections] = useState<Array<Connection>>([]);

  const handleFileSelect = async (id: string, e: Event) => {
    const file = e.target.files[0];
    const instance = nodeInstances[id];
    instance.path = URL.createObjectURL(file);
  };

  const runNode = async (id: string) => {
    const instance = nodeInstances[id];
    const outConnections = connections.filter(c => c.source === id);
    
    instance.on('processed', async (data) => {
      for (const conn of outConnections) {
        const targetInstance = nodeInstances[conn.target];
        await targetInstance.process(data);
      }
    });

    await instance.run();
  };

  const addNode = async (type: NodeType, position: Position) => {
    const id = crypto.randomUUID();
    const config: NodeConfig = type === 'source' ? {
      path: './test.txt',
      chunkSize: 100
    } : {
      process: async (input) => input
    };

    const instance = NodeFactory.createNode(type, config);
    setNodeInstances(prev => ({
      ...prev,
      [id]: instance
    }));

    const newNode: Node = {
      id,
      type,
      position,
      data: {
        content: type === 'source' ? 
          <input type="file" onChange={(e) => handleFileSelect(id, e)} /> :
          <button onClick={() => runNode(id)}>Run</button>
      }
    };

    setNodes(prev => [...prev, newNode]);
  };

  return (
    <div class="app">
      <div class="toolbar">
        <button onClick={() => addNode('source', {x: 100, y: 100})}>Add Source</button>
        <button onClick={() => addNode('processor', {x: 100, y: 200})}>Add Processor</button>
        <button onClick={() => addNode('sink', {x: 100, y: 300})}>Add Sink</button>
      </div>
      <div class="workspace">
        {nodes.map(node => (
          <NodeComponent key={node.id} {...node} />
        ))}
      </div>
    </div>
  );
} 