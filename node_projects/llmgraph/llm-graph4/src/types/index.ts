import { h } from 'preact';

export interface Position {
  x: number;
  y: number;
}

export interface Port {
  id: string;
  type: string;
  data?: unknown;
}

export type NodeType = 'llm' | 'timer' | 'memory' | 'merger' | 'output';

export interface CursorState {
  position: number;
  buffer: unknown[];
}

export interface NodeData {
  content: h.JSX.Element;
  inputs?: Port[];
  outputs?: Port[];
  cursorState?: CursorState;
  onRun?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export interface Node {
  id: string;
  type: NodeType;
  position: Position;
  data: NodeData;
}

export interface Connection {
  id: string;
  source: string;
  target: string;
}

export interface ProcessConfig {
  model?: string;
  prompt?: string;
  process: (input: unknown) => Promise<unknown>;
}

export interface TimerConfig {
  interval: number;
}

export interface MemoryConfig {
  memoryType: 'short' | 'long';
  maxSize: number;
  retrievalStrategy: 'fifo' | 'relevance';
}

export type NodeConfig = ProcessConfig | TimerConfig | MemoryConfig;
