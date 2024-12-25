const { JSDOM } = require('jsdom');

const dom = new JSDOM(`
<!DOCTYPE html>
<html>
<body>
  <div id="workspace"></div>
</body>
</html>
`);

global.window = dom.window;
global.document = dom.window.document;
global.navigator = dom.window.navigator;
global.connections = [];

// Mock LeaderLine
global.LeaderLine = class {
  constructor() {
    this.remove = jest.fn();
  }
};

// Mock WebSocket
global.WebSocket = class {
  constructor() {
    setTimeout(() => this.onopen?.(), 0);
  }
  send() {}
};

// Mock window methods
global.requestAnimationFrame = callback => setTimeout(callback, 0);
  </rewritten_file> 