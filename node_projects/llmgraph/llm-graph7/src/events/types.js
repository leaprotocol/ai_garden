/**
 * Standard event structure for all node communication
 */

export const EVENT_TYPES = {
    // Node lifecycle events
    NODE_CREATED: 'node.created',
    NODE_REMOVED: 'node.removed',
    NODE_UPDATED: 'node.updated',

    // Node output events
    NODE_OUTPUT: 'node.output',
    NODE_ERROR: 'node.error',
    NODE_COMPLETE: 'node.complete',

    // LLM specific events
    LLM_CHUNK: 'llm.chunk',
    LLM_COMPLETE: 'llm.complete',
    LLM_ERROR: 'llm.error',

    // Spawner specific events
    SPAWN_REQUEST: 'spawn.request',
    SPAWN_COMPLETE: 'spawn.complete',

    // Bucket specific events
    BUCKET_RECEIVED: 'bucket.received',
    BUCKET_LEAKED: 'bucket.leaked',
    BUCKET_FULL: 'bucket.full'
};

/**
 * Standard event structure
 * @typedef {Object} NodeEvent
 * @property {string} type - Event type from EVENT_TYPES
 * @property {string} nodeId - ID of the node emitting the event
 * @property {number} timestamp - Event timestamp
 * @property {Object} [data] - Event specific data
 * @property {string} [parentId] - Parent node ID if applicable
 * @property {Object} [metadata] - Additional metadata
 */

/**
 * Create a standard node event
 * @param {string} type - Event type from EVENT_TYPES
 * @param {string} nodeId - Node ID
 * @param {Object} data - Event data
 * @param {Object} [options] - Additional options
 * @returns {NodeEvent}
 */
export function createEvent(type, nodeId, data, options = {}) {
    return {
        type,
        nodeId,
        timestamp: Date.now(),
        data,
        parentId: options.parentId,
        metadata: options.metadata
    };
}

// Example event creators for common events
export const events = {
    // Node lifecycle
    nodeCreated: (nodeId, nodeType, config) => createEvent(EVENT_TYPES.NODE_CREATED, nodeId, {
        nodeType,
        config
    }),

    nodeRemoved: (nodeId, parentId) => createEvent(EVENT_TYPES.NODE_REMOVED, nodeId, {}, {
        parentId
    }),

    // Node output
    output: (nodeId, content) => createEvent(EVENT_TYPES.NODE_OUTPUT, nodeId, {
        content
    }),

    // LLM events
    llmChunk: (nodeId, chunk) => createEvent(EVENT_TYPES.LLM_CHUNK, nodeId, {
        chunk
    }),

    llmComplete: (nodeId) => createEvent(EVENT_TYPES.LLM_COMPLETE, nodeId),

    llmError: (nodeId, error) => createEvent(EVENT_TYPES.NODE_ERROR, nodeId, {
        error: error.message || error
    }),

    // Spawner events
    spawnRequest: (nodeId, config) => createEvent(EVENT_TYPES.SPAWN_REQUEST, nodeId, {
        nodeType: config.nodeType,
        timeout: config.timeout,
        promptData: config.promptData
    }),

    spawnComplete: (nodeId, spawnedId, config) => createEvent(EVENT_TYPES.SPAWN_COMPLETE, spawnedId, {
        config
    }, {
        parentId: nodeId
    }),

    // Bucket events
    bucketReceived: (nodeId, content) => createEvent(EVENT_TYPES.BUCKET_RECEIVED, nodeId, {
        content
    }),

    bucketLeaked: (nodeId, content) => createEvent(EVENT_TYPES.BUCKET_LEAKED, nodeId, {
        content
    }),

    bucketFull: (nodeId) => createEvent(EVENT_TYPES.BUCKET_FULL, nodeId)
}; 