import NodeFactory from './nodes/NodeFactory.js';

let nextNodeId = 1;
const nodes = {};
const connections = [];
const svg = document.getElementById('connections-svg');

// Add Node to Workspace
export function addNode(type = 'output', config = {}) {
    const id = nextNodeId++;
    const nodeInstance = NodeFactory.createNode(id, type, config);
    nodes[id] = nodeInstance;

    // Create DOM Element for Node
    const nodeElement = createNodeElement(nodeInstance);
    document.getElementById('workspace').appendChild(nodeElement);
    makeDraggable(nodeElement);

    return nodeInstance;
}

// Create Node Element in DOM
function createNodeElement(node) {
    const nodeDiv = document.createElement('div');
    nodeDiv.id = node.id;
    nodeDiv.className = 'node';
    nodeDiv.style.top = '100px';
    nodeDiv.style.left = '100px';

    const header = document.createElement('div');
    header.className = 'node-header';
    header.textContent = `${node.type.toUpperCase()} Node`;
    nodeDiv.appendChild(header);

    const content = document.createElement('div');
    content.className = 'node-content';
    content.innerHTML = getNodeContent(node);
    nodeDiv.appendChild(content);

    // Optional: Add event listeners for connecting nodes
    header.addEventListener('dblclick', () => {
        initiateConnection(node.id);
    });

    return nodeDiv;
}

// Get Node-specific Content
function getNodeContent(node) {
    switch(node.type) {
        case 'llm':
            return `
                <select class="model-select">
                    <option value="gpt-3">GPT-3</option>
                    <option value="gpt-4">GPT-4</option>
                </select>
                <textarea class="prompt" placeholder="Enter prompt here"></textarea>
            `;
        case 'timer':
            return `
                <input type="number" class="interval" value="1000" min="100">
                <button onclick="runNode(${node.id})">Run</button>
                <div class="output"></div>
            `;
        case 'memory':
            return `
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
                <button onclick="clearMemory(${node.id})">Clear Memory</button>
                <div class="memory-entries"></div>
                <div class="memory-stats">Stored: <span class="entry-count">0</span></div>
            `;
        case 'merger':
            return `
                <select class="model-select">
                    <option value="gpt-3">GPT-3</option>
                    <option value="gpt-4">GPT-4</option>
                </select>
                <textarea class="prompt" placeholder="Enter merger prompt here"></textarea>
            `;
        case 'output':
        default:
            return `
                <div class="output"></div>
            `;
    }
}

// Make Nodes Draggable
export function makeDraggable(node) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    node.querySelector('.node-header').onmousedown = dragMouseDown;

    function dragMouseDown(e) {
        e.preventDefault();
        pos3 = e.clientX;
        pos4 = e.clientY;
        document.onmouseup = closeDragElement;
        document.onmousemove = elementDrag;
    }

    function elementDrag(e) {
        e.preventDefault();
        pos1 = pos3 - e.clientX;
        pos2 = pos4 - e.clientY;
        pos3 = e.clientX;
        pos4 = e.clientY;
        node.style.top = (node.offsetTop - pos2) + "px";
        node.style.left = (node.offsetLeft - pos1) + "px";
        updateConnections();
    }

    function closeDragElement() {
        document.onmouseup = null;
        document.onmousemove = null;
    }
}

// Clear Workspace
export function clearWorkspace() {
    const workspace = document.getElementById('workspace');
    workspace.innerHTML = '';
    // Remove all SVG lines
    while (svg.firstChild) {
        svg.removeChild(svg.firstChild);
    }
    connections.length = 0;
    nextNodeId = 1;
}

// Connect Two Nodes with SVG Lines
export function connectNodes(sourceNode, targetNode) {
    const connection = { source: sourceNode, target: targetNode, line: null };
    connections.push(connection);
    drawLine(connection);
}

// Draw SVG Line Between Nodes
function drawLine(connection) {
    const sourceElement = document.getElementById(connection.source.id);
    const targetElement = document.getElementById(connection.target.id);

    const sourceRect = sourceElement.getBoundingClientRect();
    const targetRect = targetElement.getBoundingClientRect();
    const workspaceRect = document.getElementById('workspace').getBoundingClientRect();

    const startX = sourceRect.left + sourceRect.width / 2 - workspaceRect.left;
    const startY = sourceRect.top + sourceRect.height / 2 - workspaceRect.top;
    const endX = targetRect.left + targetRect.width / 2 - workspaceRect.left;
    const endY = targetRect.top + targetRect.height / 2 - workspaceRect.top;

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", startX);
    line.setAttribute("y1", startY);
    line.setAttribute("x2", endX);
    line.setAttribute("y2", endY);
    line.setAttribute("stroke", "#000");
    line.setAttribute("stroke-width", "2");
    svg.appendChild(line);

    connection.line = line;
}

// Update All SVG Lines Positions
function updateConnections() {
    connections.forEach(connection => {
        if (connection.line) {
            const sourceElement = document.getElementById(connection.source.id);
            const targetElement = document.getElementById(connection.target.id);

            const sourceRect = sourceElement.getBoundingClientRect();
            const targetRect = targetElement.getBoundingClientRect();
            const workspaceRect = document.getElementById('workspace').getBoundingClientRect();

            const startX = sourceRect.left + sourceRect.width / 2 - workspaceRect.left;
            const startY = sourceRect.top + sourceRect.height / 2 - workspaceRect.top;
            const endX = targetRect.left + targetRect.width / 2 - workspaceRect.left;
            const endY = targetRect.top + targetRect.height / 2 - workspaceRect.top;

            connection.line.setAttribute("x1", startX);
            connection.line.setAttribute("y1", startY);
            connection.line.setAttribute("x2", endX);
            connection.line.setAttribute("y2", endY);
        }
    });
}

// Remove Connection
function removeConnection(connection) {
    if (connection.line) {
        svg.removeChild(connection.line);
    }
    const index = connections.indexOf(connection);
    if (index > -1) {
        connections.splice(index, 1);
    }
}

// Save Flow to Local Storage
export function saveFlow() {
    const workspace = document.getElementById('workspace');
    const nodeElements = Array.from(workspace.getElementsByClassName('node'));
    const savedNodes = nodeElements.map(node => ({
        id: node.id,
        type: nodes[node.id].type,
        position: {
            top: node.style.top,
            left: node.style.left
        },
        config: getNodeConfig(node)
    }));

    const savedConnections = connections.map(conn => ({
        sourceId: conn.source.id,
        targetId: conn.target.id
    }));

    const flowState = {
        nodes: savedNodes,
        connections: savedConnections,
        nextNodeId
    };

    localStorage.setItem('flowState', JSON.stringify(flowState));
}

// Get Node Configuration Based on Type
function getNodeConfig(node) {
    const nodeType = nodes[node.id].type;
    let config = {};

    switch(nodeType) {
        case 'llm':
            config.model = node.querySelector('.model-select').value;
            config.prompt = node.querySelector('.prompt').value;
            break;
        case 'timer':
            config.interval = node.querySelector('.interval').value;
            break;
        case 'memory':
            config.memoryType = node.querySelector('.memory-type').value;
            config.maxSize = parseInt(node.querySelector('.memory-size').value, 10);
            config.retrievalStrategy = node.querySelector('.retrieval-strategy').value;
            break;
        case 'merger':
            config.model = node.querySelector('.model-select').value;
            config.prompt = node.querySelector('.prompt').value;
            break;
        default:
            break;
    }

    return config;
}

// Load Flow from Local Storage
export function loadFlow() {
    const stored = localStorage.getItem('flowState');
    if (!stored) return;

    const state = JSON.parse(stored);
    const workspace = document.getElementById('workspace');
    workspace.innerHTML = '';

    // Remove existing SVG lines
    while (svg.firstChild) {
        svg.removeChild(svg.firstChild);
    }

    connections.length = 0;
    Object.keys(nodes).forEach(id => delete nodes[id]);
    nextNodeId = 1;

    state.nodes.forEach(nodeData => {
        const nodeInstance = NodeFactory.createNode(nodeData.id, nodeData.type, nodeData.config);
        nodes[nodeData.id] = nodeInstance;

        const nodeElement = createNodeElement(nodeInstance);
        nodeElement.style.top = nodeData.position.top;
        nodeElement.style.left = nodeData.position.left;
        workspace.appendChild(nodeElement);
        makeDraggable(nodeElement);

        // Restore node-specific states
        if (nodeData.config.running && nodeData.type === 'timer') {
            startTimer(nodeElement);
            nodeElement.classList.add('running');
        } else if (nodeData.config.running) {
            nodeElement.classList.add('running');
        }
    });

    state.connections.forEach(connData => {
        const sourceNode = nodes[connData.sourceId];
        const targetNode = nodes[connData.targetId];
        if (sourceNode && targetNode) {
            connectNodes(sourceNode, targetNode);
        }
    });

    nextNodeId = state.nextNodeId;
}

// Function to initiate connection (optional, implement as needed)
let connectingNode = null;

function initiateConnection(nodeId) {
    if (connectingNode === null) {
        connectingNode = nodeId;
        console.log(`Selecting node ${nodeId} as source for connection.`);
    } else {
        if (connectingNode !== nodeId) {
            const sourceNode = nodes[connectingNode];
            const targetNode = nodes[nodeId];
            connectNodes(sourceNode, targetNode);
            console.log(`Connecting node ${connectingNode} to node ${nodeId}.`);
        }
        connectingNode = null;
    }
}

// Function to start timer (implement as needed)
function startTimer(nodeElement) {
    // Example implementation
    const intervalInput = nodeElement.querySelector('.interval');
    const outputDiv = nodeElement.querySelector('.output');
    const nodeId = nodeElement.id;

    nodeElement.intervalId = setInterval(() => {
        const event = { timestamp: Date.now() };
        handleEvent(nodeId, event);
    }, parseInt(intervalInput.value, 10));
}

// Function to handle events (implement as needed)
function handleEvent(nodeId, event) {
    // Your event handling logic here
    console.log(`Event received at node ${nodeId}:`, event);
}

// Function to clear memory (implement as needed)
export function clearMemory(nodeId) {
    const nodeInstance = nodes[nodeId];
    if (nodeInstance && nodeInstance.type === 'memory') {
        nodeInstance.clearMemory();
        // Update UI accordingly
        updateMemoryDisplay(document.getElementById(nodeId));
    }
}

// Function to update memory display (implement as needed)
function updateMemoryDisplay(nodeElement) {
    // Example implementation
    const memoryEntries = nodeElement.querySelector('.memory-entries');
    const entryCount = nodeElement.querySelector('.entry-count');
    // Populate memory entries and update count
}