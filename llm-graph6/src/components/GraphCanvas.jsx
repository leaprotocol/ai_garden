import React, { useImperativeHandle, forwardRef, useState, useCallback, useRef, useEffect } from 'react';
import ReactFlow, { addEdge, useEdgesState, useNodesState, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import TimerNode from './nodes/TimerNode';
import OutputNode from './nodes/OutputNode';
// Import other node types as needed

const nodeTypes = {
  timer: TimerNode,
  output: OutputNode,
  // Add other node types here
};

const GraphCanvas = forwardRef((props, ref) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  
  // Ref to keep track of the latest edges
  const edgesRef = useRef(edges);

  useEffect(() => {
    edgesRef.current = edges;
  }, [edges]);

  const handleNodeDataChange = useCallback((nodeId, newData) => {
    console.log(`GraphCanvas: handleNodeDataChange for node ${nodeId}:`, newData);

    // Update the source node's data
    setNodes((nds) => 
      nds.map((node) => 
        node.id === nodeId ? { ...node, data: { ...node.data, ...newData } } : node
      )
    );

    // Find all connected target nodes using the latest edges
    const connectedEdges = edgesRef.current.filter(edge => edge.source === nodeId);
    console.log('GraphCanvas: Connected edges:', connectedEdges);

    // Propagate the event to all connected target nodes
    connectedEdges.forEach(edge => {
      const targetNodeId = edge.target;
      if (!targetNodeId) {
        console.warn(`GraphCanvas: Connected edge has no target node.`);
        return;
      }

      // Update the target node's data with the lastEvent
      setNodes((nds) => 
        nds.map((node) => 
          node.id === targetNodeId 
            ? { ...node, data: { ...node.data, lastEvent: newData.lastEvent } } 
            : node
        )
      );
      console.log(`GraphCanvas: Propagated event to node ${targetNodeId}`);
    });
  }, []);

  const onConnect = useCallback((params) => {
    console.log('New connection:', params);
    setEdges((eds) => addEdge(params, eds));
  }, []);

  const onAddNode = useCallback((type) => {
    const id = `node_${Date.now()}`;
    const newNode = {
      id,
      type,
      position: { x: Math.random() * 250, y: Math.random() * 250 },
      data: { 
        interval: type === 'timer' ? 1000 : undefined
      },
    };

    // If it's a timer node, add onNodeDataChange handler
    if (type === 'timer') {
      newNode.data.onNodeDataChange = handleNodeDataChange;
    }

    setNodes((nds) => nds.concat(newNode));
    console.log(`Added node of type ${type} with id ${id}`);
  }, [handleNodeDataChange, setNodes]);

  const saveGraph = useCallback(() => {
    const graphState = {
      nodes: nodes.map(node => ({
        ...node,
        data: {
          ...node.data,
          onNodeDataChange: undefined // Remove function reference before saving
        }
      })),
      edges: edges
    };
    localStorage.setItem('graph-state', JSON.stringify(graphState));
    console.log('Graph state saved:', graphState);
  }, [nodes, edges]);

  const loadGraph = useCallback(() => {
    const savedState = localStorage.getItem('graph-state');
    if (savedState) {
      const { nodes: savedNodes, edges: savedEdges } = JSON.parse(savedState);
      
      // Restore onNodeDataChange for timer nodes
      const restoredNodes = savedNodes.map(node => {
        if (node.type === 'timer') {
          return {
            ...node,
            data: {
              ...node.data,
              onNodeDataChange: handleNodeDataChange
            }
          };
        }
        return node;
      });

      setNodes(restoredNodes);
      setEdges(savedEdges);
      console.log('Graph state loaded:', { nodes: restoredNodes, edges: savedEdges });
    }
  }, [handleNodeDataChange, setNodes, setEdges]);

  const clearGraph = useCallback(() => {
    setNodes([]);
    setEdges([]);
    console.log('Graph cleared');
  }, [setNodes, setEdges]);

  useImperativeHandle(ref, () => ({
    onAddNode,
    saveGraph,
    loadGraph,
    clearGraph
  }));

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      nodeTypes={nodeTypes}
      fitView
    >
      <Controls />
    </ReactFlow>
  );
});

export default GraphCanvas;