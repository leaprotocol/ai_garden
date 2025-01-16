const nodes = require('../js/nodes');

describe('Node Creation', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="workspace"></div>';
    global.connections = [];
  });

  afterEach(() => {
    jest.clearAllMocks();
    document.body.innerHTML = '';
  });

  test('creates node with correct structure', () => {
    const node = nodes.createNodeElement(1);
    expect(node.id).toBe('1');
    expect(node.className).toBe('node');
    expect(node.querySelector('.node-header')).toBeTruthy();
    expect(node.querySelector('.node-content')).toBeTruthy();
  });
}); 