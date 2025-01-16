// @flow
import { h } from 'preact';

type Props = {
  addNode: (type: string) => void,
  clearWorkspace: () => void,
  saveFlow: () => void,
  loadFlow: () => void
};

const Menu = ({ addNode, clearWorkspace, saveFlow, loadFlow }: Props) => (
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