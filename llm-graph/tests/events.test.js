const { emitNodeEvent, animateEventDot } = require('../js/main.js');

describe('Event Propagation', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="workspace"></div>';
  });

  test('events propagate to connected nodes', () => {
    const sourceNode = createNodeElement(1);
    const targetNode = createNodeElement(2);
    document.body.appendChild(sourceNode);
    document.body.appendChild(targetNode);
    
    global.connections = [{
      start: { id: '1' },
      end: { id: '2' }
    }];
    
    emitNodeEvent(sourceNode, { type: 'test', data: 'hello' });
    
    expect(targetNode.querySelector('.output').textContent)
      .toContain('hello');
  });
}); 