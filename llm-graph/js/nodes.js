const nodeTemplates = {
  header: (id) => `
    <div class="move-icon">⋮⋮</div>
    <span>Node ${id}</span>
    <div onclick="event.stopPropagation()">
      <button onclick="startConnection(${id})">Connect</button>
      <button onclick="deleteNode(${id})">×</button>
      <button onclick="toggleDebug(${id})" class="debug-btn">🐛</button>
    </div>`,

  llm: () => `
    <select class="model-select">
      <option value="">Select Model</option>
    </select>
    <textarea placeholder="Enter prompt..."></textarea>`,

  timer: () => `
    <div class="timer-controls">
      <input type="number" class="interval" value="1000" min="100" step="100">
      <label>ms interval</label>
    </div>
    <div class="output"></div>
    <div class="debug-output" style="display: none;"></div>
  `,

  output: () => `
    <div class="output"></div>
    <div class="debug-output" style="display: none;"></div>`,

  base: (id) => `
    <select class="node-type" onchange="handleNodeTypeChange(this)">
      <option value="llm">LLM Node</option>
      <option value="timer">Timer Node</option>
      <option value="output">Output Node</option>
      <option value="condition">Condition Node</option>
      <option value="memory">Memory Node</option>
      <option value="fileSystem">File System Node</option>
    </select>
    <div class="node-controls">
      <div class="llm-controls">${nodeTemplates.llm()}</div>
      <div class="timer-controls" style="display: none">${nodeTemplates.timer()}</div>
      <div class="output-controls" style="display: none">${nodeTemplates.output()}</div>
      <div class="condition-controls" style="display: none">${nodeTemplates.condition()}</div>
      <div class="memory-controls" style="display: none">${nodeTemplates.memory()}</div>
      <div class="fs-controls" style="display: none">${nodeTemplates.fileSystem()}</div>
    </div>
    <div class="output"></div>
    <button onclick="runNode(${id})">Run</button>`,

  condition: () => `
    <textarea class="condition-code" placeholder="// JavaScript condition that returns true/false
// Available variables:
// event - the incoming event object
// buffer - array of all events since last emit
// time - milliseconds since last emit
// Example: buffer.length >= 3 && time > 1000
"></textarea>
    <div class="buffer-info">
      Buffered events: <span class="buffer-count">0</span>
    </div>
  `,

  memory: () => `
    <select class="memory-type">
      <option value="short">Short-term Memory</option>
      <option value="long">Long-term Memory</option>
    </select>
    <input type="number" class="memory-size" value="10" min="1" step="1">
    <label>max entries</label>
    <select class="retrieval-strategy">
      <option value="fifo">FIFO</option>
      <option value="relevance">Relevance</option>
    </select>
    <button onclick="clearMemory(this)">Clear Memory</button>
    <div class="memory-entries"></div>
    <div class="memory-stats">Stored: <span class="entry-count">0</span></div>
  `,

  fileSystem: () => `
    <div class="fs-controls">
      <select class="source-type">
        <option value="epub">EPUB Book</option>
        <option value="photos">Photo Directory</option>
        <option value="project">Project Directory</option>
        <option value="shell">Shell Commands</option>
      </select>
      
      <input type="file" class="file-input" style="display: none">
      <input type="text" class="path-input" placeholder="/path/to/source">
      <button class="browse-btn">Browse</button>
      
      <div class="processing-options">
        <label>Chunk Size:</label>
        <input type="number" class="chunk-size" value="1000" min="100">
        
        <label>Max Depth:</label>
        <input type="number" class="max-depth" value="5" min="1">
        
        <label>Processing Mode:</label>
        <select class="process-mode">
          <option value="sequential">Sequential</option>
          <option value="parallel">Parallel</option>
          <option value="streaming">Streaming</option>
        </select>
      </div>
      
      <div class="progress-bar">
        <div class="progress"></div>
        <span class="progress-text">0%</span>
      </div>
    </div>
    <div class="output"></div>
  `
};

function createNodeElement(id) {
    const node = document.createElement('div');
    node.id = id;
    node.className = 'node';
    
    const handle = document.createElement('div');
    handle.className = 'node-header';
    handle.innerHTML = nodeTemplates.header(id);
    
    const content = document.createElement('div');
    content.className = 'node-content';
    content.innerHTML = nodeTemplates.base(id);
    
    node.appendChild(handle);
    node.appendChild(content);
    setupInputHandlers(node);
    
    return node;
}

function setupInputHandlers(node) {
    const userInput = node.querySelector('.user-input');
    if (userInput) {
        let previousValue = '';
        userInput.addEventListener('input', (e) => {
            const currentValue = e.target.value;
            emitNodeEvent(node, {
                type: 'input',
                text: currentValue.slice(previousValue.length),
                fullText: currentValue
            });
            previousValue = currentValue;
        });
    }
}

function handleNodeTypeChange(select) {
    const node = select.closest('.node');
    const controls = node.querySelectorAll('.node-controls > div');
    const selectedType = select.value;
    
    controls.forEach(control => {
        const isMatch = control.className.includes(selectedType) || 
                       (selectedType === 'fileSystem' && control.className.includes('fs'));
        control.style.display = isMatch ? 'block' : 'none';
    });
}

function handleConditionTypeChange(select) {
  const node = select.closest('.node');
  const countControls = node.querySelector('.count-controls');
  const intervalControls = node.querySelector('.interval-controls');
  const contentControls = node.querySelector('.content-controls');
  
  countControls.style.display = 'none';
  intervalControls.style.display = 'none';
  contentControls.style.display = 'none';
  
  if (select.value === 'count') {
    countControls.style.display = 'block';
  } else if (select.value === 'interval') {
    intervalControls.style.display = 'block';
  } else if (select.value === 'content') {
    contentControls.style.display = 'block';
  }
}

function evaluateCondition(node, event) {
  const conditionType = node.querySelector('.condition-type').value;
  
  switch (conditionType) {
    case 'count': {
      const count = parseInt(node.querySelector('.event-count').value);
      node.eventCount = (node.eventCount || 0) + 1;
      return node.eventCount % count === 0;
    }
    
    case 'interval': {
      const interval = parseFloat(node.querySelector('.time-interval').value) * 1000;
      const now = Date.now();
      const lastTime = node.lastEventTime || 0;
      node.lastEventTime = now;
      return (now - lastTime) >= interval;
    }
    
    case 'content': {
      const matchText = node.querySelector('.content-match').value;
      return event.chunk?.includes(matchText) || 
             event.text?.includes(matchText) ||
             event.fullText?.includes(matchText);
    }
    
    default:
      return true;
  }
} 