import { WebSocketClient } from './websocket.js';
import { EventCapture } from './eventCapture.js';

const wsClient = new WebSocketClient();
const eventCapture = new EventCapture(wsClient);
const MODEL_PREFIX = 'smollm2'; // Prefix to filter models
let selectedModel = null;

// Handle model list updates
wsClient.onMessage('available_models', data => {
    updateModelSelector(data.models);
    autoSelectModel(data.models);
});

// Handle streaming responses
wsClient.onMessage('llm_response', data => {
    wsClient.appendResponse(data.text, data.requestId);
});

// Handle errors
wsClient.onMessage('error', data => {
    wsClient.appendResponse(`Error: ${data.text}`, data.requestId, 'error');
});

function updateModelSelector(models) {
    console.log('Models received in updateModelSelector:', models);
    const modelSelect = document.getElementById('modelSelect');
    modelSelect.innerHTML = '';

    // Remove existing event listeners before adding a new one
    const newModelSelect = modelSelect.cloneNode(false);
    modelSelect.parentNode.replaceChild(newModelSelect, modelSelect);

    models.forEach(model => {
        console.log('Processing model:', model);
        const option = document.createElement('option');
        option.value = model.name;
        option.textContent = model.name;
        newModelSelect.appendChild(option);
    });

    newModelSelect.addEventListener('change', (e) => {
        selectedModel = e.target.value;
        console.log(`Model changed to: ${selectedModel}`);
        updateCurrentModelStatus(selectedModel);
    });

    const inputSection = document.querySelector('.input-section');
    if (!document.getElementById('modelSelect')) {
        inputSection.insertBefore(newModelSelect, inputSection.firstChild);
    }
}

function autoSelectModel(models) {
    const modelToSelect = models.find(model => model.name.startsWith(MODEL_PREFIX));
    if (modelToSelect) {
        selectedModel = modelToSelect.name;
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            modelSelect.value = selectedModel;
            // Trigger change event to ensure handlers are called
            modelSelect.dispatchEvent(new Event('change'));
            console.log(`Auto-selected model: ${selectedModel}`);
        }
    } else {
        console.warn(`No model found with prefix: ${MODEL_PREFIX}`);
        // Select first available model as fallback
        if (models.length > 0) {
            selectedModel = models[0].name;
            const modelSelect = document.getElementById('modelSelect');
            if (modelSelect) {
                modelSelect.value = selectedModel;
                modelSelect.dispatchEvent(new Event('change'));
                console.log(`Fallback to first available model: ${selectedModel}`);
            }
        }
    }
}

function debugLog(message, data = null) {
    console.log(`[DEBUG] ${message}`, data || '');
    const debugOutput = document.getElementById('debugOutput');
    if (debugOutput) {
        const timestamp = new Date().toLocaleTimeString();
        const logMessage = `[${timestamp}] ${message}${data ? ': ' + JSON.stringify(data, null, 2) : ''}`;
        debugOutput.textContent = logMessage + '\n' + debugOutput.textContent;
    }
}

function updateCurrentModelStatus(modelName) {
    const currentModelStatus = document.getElementById('currentModel');
    if (currentModelStatus) {
        currentModelStatus.textContent = modelName || 'None';
    }
}

// Make debugLog available globally
window.debugLog = debugLog; 