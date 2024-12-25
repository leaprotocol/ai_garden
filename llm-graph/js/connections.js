const connections = [];

    function startConnection(nodeId) {
    const sourceNode = document.getElementById(nodeId);
    const workspace = document.getElementById('workspace');
    
    document.body.style.cursor = 'crosshair';
    workspace.classList.add('connecting');
    
    const tempLine = new LeaderLine(
        sourceNode,
        LeaderLine.pointAnchor(document.body, { x: 0, y: 0 }),
        { color: '#2196F3', size: 2, dash: true }
    );
    
    const mouseMoveHandler = (e) => {
        tempLine.end = LeaderLine.pointAnchor(document.body, {
            x: e.clientX,
            y: e.clientY
        });
        
        const targetNode = e.target.closest('.node');
        document.querySelectorAll('.node').forEach(node => {
            node.classList.remove('potential-target');
        });
        if (targetNode && targetNode !== sourceNode) {
            targetNode.classList.add('potential-target');
        }
    };
    
    const clickHandler = (e) => {
        const targetNode = e.target.closest('.node');
        if (targetNode && targetNode !== sourceNode) {
            const line = new LeaderLine(sourceNode, targetNode, {
                color: '#2196F3',
                size: 2,
                path: 'straight'
            });
            
            const connection = {
                start: sourceNode,
                end: targetNode,
                startId: sourceNode.id,
                endId: targetNode.id,
                line: line,
                remove: function() {
                    this.line.remove();
                }
            };
            
            connections.push(connection);
            console.log('Created connection:', {
                startId: sourceNode.id,
                endId: targetNode.id,
                connection
            });
        }
        
        tempLine.remove();
        document.body.style.cursor = 'default';
        workspace.classList.remove('connecting');
        document.querySelectorAll('.node').forEach(node => {
            node.classList.remove('potential-target');
        });
        document.removeEventListener('mousemove', mouseMoveHandler);
        document.removeEventListener('click', clickHandler);
    };
    
    document.addEventListener('mousemove', mouseMoveHandler);
    document.addEventListener('click', clickHandler);
}

function updateConnections() {
    connections.forEach(connection => {
        if (connection.line && connection.line.position) {
            connection.line.position();
        } else if (connection.line) {
            // Recreate the line with the same properties
            const newLine = new LeaderLine(
                connection.start,
                connection.end,
                { color: '#2196F3', size: 2, path: 'straight' }
            );
            
            // Remove old line
            connection.line.remove();
            
            // Update connection with new line
            connection.line = newLine;
        }
    });
} 