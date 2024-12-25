export class EventSystem {
  constructor() {
    this.ws = new WebSocket(`ws://${window.location.hostname}:3000`);
    this.modelRefreshRetries = 0;
    this.MAX_RETRIES = 3;
    this.setupWebSocket();
    this.timers = new Map(); // Track running timers
  }

  setupWebSocket() {
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.refreshModels();
      this.requestNodes();
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'nodesUpdated':
          this.renderGraph(data.nodes, data.connections);
          break;
        case 'models':
          this.handleModels(data.models);
          break;
        case 'nodeEvent':
          this.handleNodeEvent(data);
          break;
        case 'toggleTimer':
          const nodeId = data.nodeId;
          const node = this.nodes.get(nodeId);
          if (node && node.type === 'timer') {
            if (this.nodeIntervals.has(nodeId)) {
              clearInterval(this.nodeIntervals.get(nodeId));
              this.nodeIntervals.delete(nodeId);
            } else {
              const interval = setInterval(() => {
                this.broadcast({
                  type: 'nodeEvent',
                  target: nodeId,
                  payload: { timestamp: Date.now() }
                });
              }, node.interval || 1000);
              this.nodeIntervals.set(nodeId, interval);
            }
          }
          break;
        case 'updateInterval':
          if (this.nodes.has(data.nodeId)) {
            const node = this.nodes.get(data.nodeId);
            node.interval = parseInt(data.interval);
            this.nodes.set(data.nodeId, node);
            // Restart timer if running
            if (this.nodeIntervals.has(data.nodeId)) {
              clearInterval(this.nodeIntervals.get(data.nodeId));
              this.nodeIntervals.set(data.nodeId, setInterval(() => {
                this.broadcast({
                  type: 'nodeEvent',
                  target: data.nodeId,
                  payload: { timestamp: Date.now() }
                });
              }, node.interval));
            }
          }
          break;
      }
    };
  }

  async refreshModels() {
    if (this.modelRefreshRetries >= this.MAX_RETRIES) {
      console.error('Max model refresh retries reached');
      return;
    }
    
    try {
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'getModels' }));
        this.modelRefreshRetries++;
      } else {
        throw new Error('WebSocket not connected');
      }
    } catch (error) {
      console.error('Error refreshing models:', error);
      setTimeout(() => this.refreshModels(), 1000);
    }
  }

  handleModels(models) {
    localStorage.setItem('availableModels', JSON.stringify(models));
    console.log('Models updated:', models);
  }

  requestNodes() {
    this.ws.send(JSON.stringify({ type: 'getNodes' }));
  }

  createNode(type, position = { x: 100, y: 100 }) {
    this.ws.send(JSON.stringify({
      type: 'createNode',
      nodeType: type,
      position
    }));
  }

  renderGraph(nodes, connections) {
    const graphDiv = document.getElementById('mermaidGraph');
    graphDiv.innerHTML = '';
    
    if (Object.keys(nodes).length === 0) {
      const emptyGraph = 'graph LR;';
      const tempDiv = document.createElement('div');
      tempDiv.className = 'mermaid';
      tempDiv.textContent = emptyGraph;
      graphDiv.appendChild(tempDiv);
      mermaid.init(undefined, tempDiv);
      return;
    }

    let graphDefinition = [
      'graph LR',
      '',
      '  %% Node Definitions',
      ...Object.entries(nodes).map(([id, node]) => {
        let controls = '';
        if (node.type === 'timer') {
          controls = `<br>
            <button onclick='window.eventSystem.toggleTimer("${id}")'>Start/Stop</button>
            <input type='number' value='1000' 
              onchange='window.eventSystem.updateInterval("${id}", this.value)'
            />ms`;
        }
        return `  ${id}["${node.type} ${id.slice(0,4)}${controls}"]`
      }),
      '',
      '  %% Connections',
      ...Object.entries(connections).map(([id, conn]) => 
        `  ${conn.source} --> ${conn.target}`
      ),
      '',
      '  %% Styles',
      '  classDef llm fill:#e1f5fe,stroke:#0288d1,stroke-width:2px',
      '  classDef timer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px',
      '  classDef output fill:#fff,stroke:#4CAF50,stroke-width:2px',
      ...Object.entries(nodes).map(([id, node]) => 
        `  class ${id} ${node.type}`
      )
    ].join('\n');

    try {
      const tempDiv = document.createElement('div');
      tempDiv.className = 'mermaid';
      tempDiv.textContent = graphDefinition;
      graphDiv.appendChild(tempDiv);
      mermaid.init(undefined, tempDiv).then(() => {
        // Make nodes draggable after rendering
        const nodes = graphDiv.querySelectorAll('.node');
        nodes.forEach(node => makeDraggable(node));
      });
    } catch (error) {
      console.error('Mermaid render error:', error);
      console.log('Graph definition:', graphDefinition);
    }
  }

  connectNodes(sourceId, targetId) {
    this.ws.send(JSON.stringify({
      type: 'connectNodes',
      source: sourceId,
      target: targetId
    }));
  }

  clearGraph() {
    this.ws.send(JSON.stringify({ 
      type: 'clearGraph' 
    }));
  }

  saveGraph() {
    this.ws.send(JSON.stringify({ type: 'saveGraph' }));
  }

  loadGraph() {
    this.ws.send(JSON.stringify({ type: 'loadGraph' }));
  }

  handleNodeEvent(event) {
    const nodeElement = document.querySelector(`#mermaidGraph #${event.target}`);
    if (nodeElement && event.payload) {
      const textElement = nodeElement.querySelector('text');
      if (textElement) {
        // Update only the timestamp part
        const existingText = textElement.textContent;
        const timestampStr = new Date(event.payload.timestamp).toLocaleTimeString();
        if (existingText.includes('Last:')) {
          textElement.textContent = existingText.replace(/Last:.*$/, `Last: ${timestampStr}`);
        } else {
          textElement.textContent += `\nLast: ${timestampStr}`;
        }
      }
    }
  }

  updateNodePosition(id, position) {
    this.ws.send(JSON.stringify({
      type: 'updateNodePosition',
      id: id,
      position: position
    }));
  }

  toggleTimer(nodeId) {
    this.ws.send(JSON.stringify({
      type: 'toggleTimer',
      nodeId: nodeId
    }));
  }

  updateInterval(nodeId, interval) {
    this.ws.send(JSON.stringify({
      type: 'updateInterval',
      nodeId: nodeId,
      interval: parseInt(interval)
    }));
  }
} 

mermaid.initialize({ 
  startOnLoad: false,
  theme: 'default',
  flowchart: {
    curve: 'basis',
    nodeSpacing: 50,
    rankSpacing: 50,
    defaultRenderer: 'dagre'
  },
  securityLevel: 'loose'
}); 

function makeDraggable(element) {
  let dragElement = element;
  if (element.tagName !== 'g') {
    dragElement = element.closest('g');
    if (!dragElement) return;
  }

  let transform = dragElement.transform.baseVal.getItem(0);
  if (!transform) {
    transform = dragElement.ownerSVGElement.createSVGTransform();
    dragElement.transform.baseVal.appendItem(transform);
  }

  let startX, startY, initialTranslateX, initialTranslateY;

  dragElement.addEventListener('mousedown', (e) => {
    startX = e.clientX;
    startY = e.clientY;
    initialTranslateX = transform.matrix.e;
    initialTranslateY = transform.matrix.f;

    document.addEventListener('mousemove', dragMove);
    document.addEventListener('mouseup', dragEnd);
  });

  function dragMove(e) {
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    transform.setTranslate(initialTranslateX + dx, initialTranslateY + dy);
  }

  function dragEnd() {
    document.removeEventListener('mousemove', dragMove);
    document.removeEventListener('mouseup', dragEnd);

    // Send updated position to server
    const nodeId = dragElement.id.split('-')[1]; // Extract node ID
    if (window.eventSystem && nodeId) {
      window.eventSystem.updateNodePosition(nodeId, {
        x: transform.matrix.e,
        y: transform.matrix.f
      });
    }
  }
} 

window.eventSystem = new EventSystem();
window.createNode = (type) => eventSystem.createNode(type);
window.clearGraph = () => eventSystem.clearGraph();
window.saveGraph = () => eventSystem.saveGraph();
window.loadGraph = () => eventSystem.loadGraph();
window.toggleTimer = (id) => eventSystem.toggleTimer(id);
window.updateInterval = (id, value) => eventSystem.updateInterval(id, value); 