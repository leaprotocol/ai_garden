// Initialize WebSocket connection
const ws = new WebSocket(`ws://${window.location.host}/ws`);
let tokenTree = null;
const MAX_TOKENS = 5;
let generatedTokenCount = 0;

// ASCII art elements
const VERTICAL = '│';
const HORIZONTAL = '─';
const TEE_RIGHT = '├';
const CORNER_RIGHT = '└';
const TEE_DOWN = '┬';
const CORNER_DOWN = '┴';

class TokenNode {
    constructor(token, probability, isAlternative = false) {
        this.token = token;
        this.probability = probability;
        this.isAlternative = isAlternative;
        this.children = [];
        this.length = 0;
        this.updateLength();
    }

    updateLength() {
        if (this.children.length === 0) {
            this.length = this.token.length;
        } else {
            // Sum of children lengths + spaces between them
            this.length = this.children.reduce((sum, child) => sum + child.length, 0) + 
                         (this.children.length - 1) * 1;
        }
    }

    addChild(child) {
        this.children.push(child);
        this.updateLength();
    }
}

function renderTree(node) {
    const lines = [];
    
    // Current node
    lines.push(node.token.padEnd(node.length));
    
    // Children in one row
    if (node.children.length > 0) {
        const childrenLine = node.children.map(child => child.token.padEnd(child.length)).join(' ');
        lines.push(childrenLine);
        
        // Render grandchildren
        node.children.forEach(child => {
            if (child.children.length > 0) {
                lines.push(...renderTree(child).slice(1));
            }
        });
    }
    
    return lines;
}

function logTreeState(node) {
    const treeState = {
        token: node.token,
        probability: `${(node.probability * 100).toFixed(0)}%`,
        length: node.length,
        children: node.children.map(child => logTreeState(child))
    };
    console.log(JSON.stringify(treeState, null, 2));
    return treeState;
}

function renderVisualization() {
    if (!tokenTree) return;
    
    const lines = renderTree(tokenTree);
    const viz = document.getElementById('visualization');
    viz.innerHTML = lines.join('\n');
    
    logTreeState(tokenTree);
}

// WebSocket message handling
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'analysis') {
        // Create initial token tree
        let currentNode = null;
        data.tokens.forEach((token, i) => {
            const newNode = new TokenNode(token.token, token.probability);
            
            // Add alternatives as children
            token.top_predictions.slice(0, 5).forEach(pred => {
                if (pred.token !== token.token) {
                    const altNode = new TokenNode(pred.token, pred.probability, true);
                    newNode.addChild(altNode);
                }
            });
            
            if (!tokenTree) {
                tokenTree = newNode;
                currentNode = newNode;
            } else {
                currentNode.addChild(newNode);
                currentNode = newNode;
            }
        });
        
        renderVisualization();
    }
    else if (data.type === 'generation') {
        if (generatedTokenCount >= MAX_TOKENS) {
            ws.send(JSON.stringify({ type: 'stop_generation' }));
            return;
        }

        const newNode = new TokenNode(data.token.token, data.token.probability);
        
        // Add alternatives
        data.token.top_predictions.slice(0, 5).forEach(pred => {
            if (pred.token !== data.token.token) {
                const altNode = new TokenNode(pred.token, pred.probability, true);
                newNode.addChild(altNode);
            }
        });
        
        // Find the last non-alternative node to attach to
        let current = tokenTree;
        while (current.children.length > 0 && !current.children[0].isAlternative) {
            current = current.children[0];
        }
        current.addChild(newNode);
        
        generatedTokenCount++;
        renderVisualization();
    }
};

// Button event handlers
document.getElementById('analyze').addEventListener('click', () => {
    const text = document.getElementById('prompt').value;
    tokenTree = null;
    ws.send(JSON.stringify({
        type: 'analyze',
        text: text
    }));
});

document.getElementById('generate').addEventListener('click', () => {
    const prompt = document.getElementById('prompt').value;
    generatedTokenCount = 0;
    tokenTree = null;
    ws.send(JSON.stringify({
        type: 'generate',
        prompt: prompt
    }));
}); 