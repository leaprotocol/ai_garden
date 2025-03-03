// Initialize WebSocket connection
const ws = new WebSocket(`ws://${window.location.host}/ws`);
let treeData = { name: "Root", children: [] };
let currentNode = treeData;

// D3 visualization setup
const margin = { top: 20, right: 90, bottom: 20, left: 90 };
const width = 960 - margin.left - margin.right;
const height = 800 - margin.top - margin.bottom;

// Create the SVG container
const svg = d3.select("#tree-container")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

// Modify the force simulation for left-to-right layout
const simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-300))
    // Remove center force to allow left-to-right layout
    .force("x", d3.forceX(d => d.depth * 180).strength(0.5)) // Horizontal positioning based on depth
    .force("y", d3.forceY(height / 2).strength(0.1)) // Keep vertical centering
    .force("collision", d3.forceCollide().radius(30)); // Prevent overlap

// Create a tooltip
const tooltip = d3.select("body")
    .append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

// Add a counter for generated tokens
let generatedTokenCount = 0;
const MAX_TOKENS = 5;

// Initialize variables for node tracking
let nodes = [];
let links = [];
let nodeId = 0;

function updateTree(source) {
    // Reset nodes and links arrays
    nodes = [];
    links = [];
    nodeId = 0;

    // Process nodes starting from root
    processNode(source);

    // Update links
    const link = svg.selectAll(".link")
        .data(links, d => d.target.id);

    link.exit().remove();

    const linkEnter = link.enter()
        .append("path")
        .attr("class", "link")
        .attr("stroke", "#ccc")
        .attr("stroke-width", 1)
        .attr("fill", "none");

    // Update nodes
    const node = svg.selectAll(".node")
        .data(nodes, d => d.id);

    node.exit().remove();

    const nodeEnter = node.enter()
        .append("g")
        .attr("class", "node")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
        .on("click", handleNodeClick);

    // Add circles for nodes
    nodeEnter.append("circle")
        .attr("r", d => {
            if (d.isInput) return 10;
            if (d.isAlternative) return 6;
            return 8;
        })
        .style("fill", d => {
            if (d.isInput) return "#e8f5e9";
            if (d.isAlternative) return "#f8f8f8";
            return "#fff";
        })
        .style("stroke", d => {
            if (d.isInput) return "#2e7d32";
            if (d.isAlternative) return "#999";
            return "#4CAF50";
        })
        .style("stroke-dasharray", d => d.isAlternative ? "2,2" : "none");

    // Add labels
    nodeEnter.append("text")
        .attr("dy", ".35em")
        .attr("x", 12)
        .text(d => d.name)
        .on("mouseover", (event, d) => {
            if (d.probability) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`Probability: ${(d.probability * 100).toFixed(2)}%`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }
        })
        .on("mouseout", () => {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        });

    // Update simulation
    simulation.nodes(nodes).on("tick", ticked);
    simulation.force("link").links(links);

    // Reheat the simulation
    simulation.alpha(1).restart();
}

// Add drag functions
function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// Modify the click handler to maintain context
function handleNodeClick(event, d) {
    if (d.isAlternative) return;
    event.stopPropagation();

    // Build the context by traversing up the tree
    let context = [];
    let current = d;
    
    // Find the path from root to this node
    while (current) {
        if (current.name !== "Start") {
            context.unshift(current.name);
        }
        // Find parent by searching through all nodes
        current = nodes.find(n => 
            n.children && n.children.some(child => child.id === current.id)
        );
    }

    // Reset generation count but keep the existing tree
    generatedTokenCount = 0;
    currentNode = d;

    // Start generation with full context
    ws.send(JSON.stringify({
        type: 'generate',
        prompt: context.join('')
    }));
}

// Modify the node processing to track depth
function processNode(node, parent = null, depth = 0) {
    const graphNode = {
        id: nodeId++,
        name: node.name,
        probability: node.probability,
        isInput: node.isInput,
        isAlternative: node.isAlternative,
        depth: depth, // Add depth information
        x: node.x || (depth * 180), // Initial x position based on depth
        y: node.y || (height / 2 + (Math.random() - 0.5) * 100), // Slight vertical variation
        vx: node.vx,
        vy: node.vy
    };
    nodes.push(graphNode);
    
    if (parent) {
        links.push({
            source: parent.id,
            target: graphNode.id,
            value: 1
        });
    }
    
    if (node.children) {
        node.children.forEach(child => 
            processNode(child, graphNode, depth + 1)
        );
    }
}

// Modify the ticked function for better positioning
function ticked() {
    // Keep nodes within bounds and maintain left-to-right layout
    nodes.forEach(d => {
        // X position based on depth with some flexibility
        d.x = Math.max(50, Math.min(width - 50, d.x));
        // Y position with more restricted movement
        d.y = Math.max(50, Math.min(height - 50, d.y));
        
        // Pull nodes towards their depth-based x position
        d.x += (d.depth * 180 - d.x) * 0.1;
    });

    // Update link positions
    svg.selectAll(".link")
        .attr("d", d => {
            const dx = d.target.x - d.source.x,
                  dy = d.target.y - d.source.y;
            return `M${d.source.x},${d.source.y}L${d.target.x},${d.target.y}`;
        });

    // Update node positions
    svg.selectAll(".node")
        .attr("transform", d => `translate(${d.x},${d.y})`);
}

// Modify the WebSocket message handling for analysis
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'analysis') {
        // Create a chain of tokens
        let currentParent = { name: "Start", depth: 0 };
        treeData = currentParent;
        
        // Chain the tokens one after another
        data.tokens.forEach((token, index) => {
            const newNode = {
                name: token.token,
                probability: token.probability,
                isInput: true,
                depth: index + 1,
                children: token.top_predictions.slice(0, 5).map(pred => ({
                    name: pred.token,
                    probability: pred.probability,
                    isAlternative: true,
                    depth: index + 2
                }))
            };
            
            currentParent.children = [newNode];
            currentParent = newNode;
        });
        
        currentNode = currentParent;
        updateTree(treeData);
    }
    else if (data.type === 'generation') {
        if (generatedTokenCount >= MAX_TOKENS) {
            ws.send(JSON.stringify({ type: 'stop_generation' }));
            return;
        }

        const newNode = {
            name: data.token.token,
            probability: data.token.probability,
            depth: currentNode.depth + 1,
            children: data.token.top_predictions.slice(0, 5).map(pred => ({
                name: pred.token,
                probability: pred.probability,
                isAlternative: true,
                depth: currentNode.depth + 2
            }))
        };
        
        if (!currentNode.children) {
            currentNode.children = [];
        }
        currentNode.children.push(newNode);
        currentNode = newNode;
        generatedTokenCount++;
        
        updateTree(treeData);
    }
};

// Button event handlers
document.getElementById('analyze').addEventListener('click', () => {
    const text = document.getElementById('prompt').value;
    ws.send(JSON.stringify({
        type: 'analyze',
        text: text
    }));
});

document.getElementById('generate').addEventListener('click', () => {
    const prompt = document.getElementById('prompt').value;
    generatedTokenCount = 0;
    currentNode = treeData;
    ws.send(JSON.stringify({
        type: 'generate',
        prompt: prompt
    }));
});

// Initial tree render
updateTree(treeData); 