// @flow
import { h } from 'preact';
import Node from './Node';

type NodeType = 'llm' | 'timer' | 'memory' | 'merger' | 'output';

type NodeConfig = {
  [key: string]: any
};

type NodeItem = {
  id: number,
  type: NodeType,
  config: NodeConfig,
  position: {
    top: string,
    left: string
  }
};

type Props = {
  nodes: Array<NodeItem>,
  connections: Array<{ sourceId: number, targetId: number }>,
  setConnections: (connections: Array<{ sourceId: number, targetId: number }>) => void
};

const Workspace = ({ nodes, connections, setConnections }: Props) => (
  <div id="workspace">
    {nodes.map(node => (
      <Node key={node.id} node={node} />
    ))}
    {/* SVG for connections can be managed here */}
  </div>
);

export default Workspace; 