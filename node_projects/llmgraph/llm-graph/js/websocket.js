let ws;
let modelRefreshRetries = 0;
const MAX_RETRIES = 3;

function connectWebSocket() {
    ws = new WebSocket('ws://localhost:3000');
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        refreshModels();
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected, retrying in 5s...');
        setTimeout(connectWebSocket, 5000);
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const node = document.getElementById(data.nodeId);
        
        if (data.type === 'models') {
            updateModelSelects(data.models);
            return;
        }
        
        if (node) {
            handleNodeMessage(node, data);
        }
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'nodeStream':
            const node = document.getElementById(data.nodeId);
            if (node) {
                const output = node.querySelector('.output');
                if (data.chunk) {
                    output.textContent += data.chunk;
                    output.scrollTop = output.scrollHeight;
                }

                // If it's not an output node, emit the event
                if (node.querySelector('.node-type').value !== 'output') {
                    emitNodeEvent(node, {
                        type: 'llm',
                        chunk: data.chunk,
                        complete: false
                    });
                }
            }
            break;

        case 'nodeComplete':
            const completedNode = document.getElementById(data.nodeId);
            if (completedNode) {
                // If it's not an output node, emit the event
                if (completedNode.querySelector('.node-type').value !== 'output') {
                    emitNodeEvent(completedNode, {
                        type: 'llm',
                        complete: true
                    });
                }
            }
            break;

        case 'error':
            const errorNode = document.getElementById(data.nodeId);
            if (errorNode) {
                const output = errorNode.querySelector('.output');
                output.textContent += `\nError: ${data.message}\n`;
                errorNode.classList.remove('running');
            }
            break;

        case 'progress':
            const progressNode = document.getElementById(data.nodeId);
            if (progressNode) {
                const progressBar = progressNode.querySelector('.progress-bar');
                const progress = progressBar.querySelector('.progress');
                const progressText = progressBar.querySelector('.progress-text');
                progressBar.style.display = 'block';
                progress.style.width = `${data.progress}%`;
                progressText.textContent = `${data.progress}%`;
                if (data.progress === 100) {
                    setTimeout(() => {
                        progressBar.style.display = 'none';
                        progress.style.width = '0%';
                        progressText.textContent = '0%';
                    }, 1000);
                }
            }
            break;

        case 'models':
            const modelSelects = document.querySelectorAll('.model-select');
            modelSelects.forEach(select => {
                data.models.forEach(model => {
                    const optionExists = Array.from(select.options).some(option => option.value === model.name);
                    if (!optionExists) {
                        const option = document.createElement('option');
                        option.value = model.name;
                        option.textContent = model.name;
                        select.appendChild(option);
                    }
                });
            });
            break;
    }
}

function handleNodeMessage(node, data) {
    const output = node.querySelector('.output');
    const progressBar = node.querySelector('.progress-bar');
    const nodeType = node.querySelector('.node-type').value;
    
    console.log('handleNodeMessage:', { nodeId: node.id, data });

    if (nodeType === 'output' && !node.outputEnabled) {
        return;
    }

    switch (data.type) {
        case 'nodeStream':
            if (data.chunk) {
                const textNode = document.createTextNode(data.chunk);
                output.appendChild(textNode);
                output.scrollTop = output.scrollHeight;
            }
            
            if (nodeType !== 'output') {
                emitNodeEvent(node, {
                    type: 'llm',
                    chunk: data.chunk,
                    complete: false
                });
            }
            break;
            
        case 'nodeComplete':
            if (data.output) {
                const textNode = document.createTextNode(data.output);
                output.appendChild(textNode);
            }
            const completedText = document.createTextNode('\nCompleted\n');
            output.appendChild(completedText);
            output.scrollTop = output.scrollHeight;
            node.classList.remove('running');
            
            if (nodeType !== 'output') {
                emitNodeEvent(node, {
                    type: 'llm',
                    output: data.output,
                    complete: true
                });
            }
            break;
            
        case 'error':
            const errorText = document.createTextNode(`\nError: ${data.error}\n`);
            output.appendChild(errorText);
            node.classList.remove('running');
            break;
    }
}

async function refreshModels() {
    if (modelRefreshRetries >= MAX_RETRIES) {
        console.error('Max model refresh retries reached');
        return;
    }
    
    try {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'getModels' }));
            modelRefreshRetries++;
        } else {
            throw new Error('WebSocket not connected');
        }
    } catch (error) {
        console.error('Error refreshing models:', error);
        setTimeout(refreshModels, 1000);
    }
}

function updateModelSelects(models) {
    localStorage.setItem('availableModels', JSON.stringify(models));
    
    document.querySelectorAll('.model-select').forEach(select => {
        const currentValue = select.value;
        select.innerHTML = '<option value="">Select Model</option>';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.name;
            select.appendChild(option);
        });
        select.value = currentValue;
    });
}

function restoreModels() {
    const savedModels = localStorage.getItem('availableModels');
    if (savedModels) {
        console.log('Restoring saved models');
        updateModelSelects(JSON.parse(savedModels));
    }
}

connectWebSocket();

window.addEventListener('load', connectWebSocket);

window.addEventListener('load', () => {
    loadFlow();
    restoreModels();
}); 