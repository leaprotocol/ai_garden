import { h } from 'preact';
import Node from './Node';
import { Node as NodeType, Connection } from '../types';

interface WorkspaceProps {
  nodes: NodeType[];
  connections: Connection[];
  setConnections: (connections: Connection[]) => void;
  onPositionChange?: (id: string, position: { x: number, y: number }) => void;
}

export const Workspace = ({ nodes, connections, setConnections, onPositionChange }: WorkspaceProps) => {
  return (
    <div id="workspace">
      {nodes.map(node => (
        <Node 
          key={node.id} 
          {...node} 
          onPositionChange={onPositionChange}
        />
      ))}
      {/* SVG for connections will be implemented here */}
    </div>
  );
};

export default Workspace;
