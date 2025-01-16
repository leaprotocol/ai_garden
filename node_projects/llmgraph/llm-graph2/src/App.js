// @flow
import { h } from 'preact';
import { useState } from 'preact/hooks';
import Menu from './Menu';
import Workspace from './Workspace';

type NodeType = 'llm' | 'timer' | 'memory' | 'merger' | 'output';

type NodeConfig = {
  [key: string]: any
};

type Node = {
  id: number,
  type: NodeType,
  config: NodeConfig,
  position: {
    top: string,
    left: string
  }
};

const App = () => {
  const [nodes, setNodes] = useState<Array<Node>>([]);
  const [connections, setConnections] = useState<Array<{ sourceId: number, targetId: number }>>([]);
  const [nextNodeId, setNextNodeId] = useState<number>(1);

  const addNode = (type: NodeType) => {
    const newNode: Node = {
      id: nextNodeId,
      type,
      config: {},
      position: { top: '100px', left: '100px' }
    };
    setNodes([...nodes, newNode]);
    setNextNodeId(nextNodeId + 1);
  };

  const clearWorkspace = () => {
    setNodes([]);
    setConnections([]);
    setNextNodeId(1);
  };

  const saveFlow = () => {
    const flowState = {
      nodes,
      connections,
      nextNodeId
    };
    localStorage.setItem('flowState', JSON.stringify(flowState));
  };

  const loadFlow = () => {
    const stored = localStorage.getItem('flowState');
    if (stored) {
      const { nodes, connections, nextNodeId } = JSON.parse(stored);
      setNodes(nodes);
      setConnections(connections);
      setNextNodeId(nextNodeId);
    }
  };

  return (
    <div>
      <Menu addNode={addNode} clearWorkspace={clearWorkspace} saveFlow={saveFlow} loadFlow={loadFlow} />
      <Workspace nodes={nodes} connections={connections} setConnections={setConnections} />
    </div>
  );
};

export default App; 