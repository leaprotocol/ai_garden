// @flow
export type Position = {|
  x: number,
  y: number
|};

export type Port = {|
  id: string,
  type: string,
  data?: any
|};

export type NodeType = 'source' | 'processor' | 'sink';

export type NodeData = {|
  content: any,
  inputs?: Array<Port>,
  outputs?: Array<Port>,
  cursorState?: {
    position: number,
    buffer: Array<any>
  },
  onRun?: (id: string) => void,
  onDelete?: (id: string) => void
|};

export type Node = {|
  id: string,
  type: NodeType,
  position: Position,
  data: NodeData
|};

export type Connection = {|
  id: string,
  source: string,
  target: string
|};

export type ProcessConfig = {|
  process: (input: any) => Promise<any>
|};

export type SourceConfig = {|
  path: string,
  chunkSize: number
|};

export type NodeConfig = ProcessConfig | SourceConfig; 