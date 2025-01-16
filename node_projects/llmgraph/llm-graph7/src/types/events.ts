// Event type definitions
export interface Event {
  id: string;          // Unique event ID (e.g., "evt_1234")
  type: string;        // Event type (e.g., "chunk", "complete", "error")
  source: string;      // Source node ID
  data: any;          // The actual payload
  meta?: {            // Optional metadata
    parent?: string;  // Parent event ID for relationships
    seq?: number;     // Sequence number for ordering
    done?: boolean;   // Completion marker
  }
}

// Event type constants
export const EventTypes = {
  CHUNK: 'chunk',
  COMPLETE: 'complete',
  ERROR: 'error',
  SPAWN: 'spawn',
  REMOVE: 'remove',
  TICK: 'tick',
  STATE: 'state'
} as const;

// Helper function to create events
export function createEvent(
  type: string,
  source: string,
  data: any,
  meta?: Event['meta']
): Event {
  return {
    id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    type,
    source,
    data,
    meta
  };
}

// Helper to create chunk events
export function createChunkEvent(
  source: string,
  data: any,
  seq?: number,
  parent?: string
): Event {
  return createEvent(EventTypes.CHUNK, source, data, {
    seq,
    parent,
    done: false
  });
}

// Helper to create completion events
export function createCompleteEvent(
  source: string,
  data: any = null,
  parent?: string
): Event {
  return createEvent(EventTypes.COMPLETE, source, data, {
    parent,
    done: true
  });
}

// Helper to create error events
export function createErrorEvent(
  source: string,
  error: string | Error
): Event {
  const errorMessage = error instanceof Error ? error.message : error;
  return createEvent(EventTypes.ERROR, source, { message: errorMessage });
}

// Type guard to check if an object is an Event
export function isEvent(obj: any): obj is Event {
  return (
    obj &&
    typeof obj.id === 'string' &&
    typeof obj.type === 'string' &&
    typeof obj.source === 'string' &&
    'data' in obj
  );
} 