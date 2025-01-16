import React, { useImperativeHandle, forwardRef, useState, useCallback, useRef, useEffect } from 'react';
import ReactFlow, { addEdge, useEdgesState, useNodesState, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import AbstractNode from './nodes/AbstractNode.jsx';
import TimerNode from './nodes/TimerNode.jsx';
import OutputNode from './nodes/OutputNode.jsx';
import { logger } from '../utils/logger.js';

const nodeTypes = {
  timer: TimerNode,
  output: OutputNode,
};

const GraphCanvas = forwardRef((props, ref) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  
  const edgesRef = useRef(edges);

  useEffect(() => {
    edgesRef.current = edges;
  }, [edges]);

  const handleNodeDataChange = useCallback((nodeId, newData) => {
    logger.info(`GraphCanvas: handleNodeDataChange for node ${nodeId}:`, newData);

    setNodes((nds) => 
      nds.map((node) => 
        node.id === nodeId ? { ...node, data: { ...node.data, ...newData } } : node
      )
    );

    const connectedEdges = edgesRef.current.filter(edge => edge.source === nodeId);
    logger.info('GraphCanvas: Connected edges:', connectedEdges);

    connectedEdges.forEach(edge => {
      const targetNodeId = edge.target;
      if (!targetNodeId) {
        logger.warn(`GraphCanvas: Connected edge has no target node.`);
        return;
      }

      setNodes((nds) => 
        nds.map((node) => 
          node.id === targetNodeId 
            ? { ...node, data: { ...node.data, lastEvent: newData.lastEvent } } 
            : node
        )
      );
      logger.info(`GraphCanvas: Propagated event to node ${targetNodeId}`);
    });
  }, []);

  const onConnect = useCallback((params) => {
    logger.info('New connection:', params);
    setEdges((eds) => addEdge(params, eds));
  }, []);

  const onAddNode = useCallback((type) => {
    const id = `node_${Date.now()}`;
    const newNode = {
      id,
      type,
      position: { x: Math.random() * 250, y: Math.random() * 250 },
      data: { 
        label: `${type.charAt(0).toUpperCase() + type.slice(1)} Node`,
        onNodeDataChange: handleNodeDataChange,
      },
    };

    setNodes((nds) => nds.concat(newNode));
    logger.info(`Added node of type ${type} with id ${id}`);
  }, [handleNodeDataChange, setNodes]);

  const saveGraph = useCallback(() => {
    const graphState = {
      nodes,
      edges
    };
    localStorage.setItem('graph-state', JSON.stringify(graphState));
    logger.info('Graph state saved:', graphState);
  }, [nodes, edges]);

  const loadGraph = useCallback(() => {
    const savedState = localStorage.getItem('graph-state');
    if (savedState) {
      const { nodes: savedNodes, edges: savedEdges } = JSON.parse(savedState);
      
      const restoredNodes = savedNodes.map(node => {
        if (node.type === 'timer' || node.type === 'output') {
          return {
            ...node,
            data: {
              ...node.data,
              onNodeDataChange: handleNodeDataChange,
            }
          };
        }
        return node;
      });

      setNodes(restoredNodes);
      setEdges(savedEdges);
      logger.info('Graph state loaded:', { nodes: restoredNodes, edges: savedEdges });
    }
  }, [handleNodeDataChange, setNodes, setEdges]);

  const clearGraph = useCallback(() => {
    setNodes([]);
    setEdges([]);
    logger.info('Graph cleared');
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