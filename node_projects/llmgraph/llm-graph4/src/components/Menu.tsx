import { h } from 'preact';
import { NodeType } from '../types';

interface MenuProps {
  addNode: (type: NodeType) => void;
  clearWorkspace: () => void;
  saveFlow: () => void;
  loadFlow: () => void;
}

export const Menu = ({ addNode, clearWorkspace, saveFlow, loadFlow }: MenuProps) => (
  <div id="menu">
    <button onClick={() => addNode('llm')}>Add LLM Node</button>
    <button onClick={() => addNode('timer')}>Add Timer Node</button>
    <button onClick={() => addNode('memory')}>Add Memory Node</button>
    <button onClick={() => addNode('merger')}>Add Merger Node</button>
    <button onClick={() => addNode('output')}>Add Output Node</button>
    <button onClick={clearWorkspace}>Clear All</button>
    <button onClick={saveFlow}>Save Flow</button>
    <button onClick={loadFlow}>Load Flow</button>
  </div>
);

export default Menu;
