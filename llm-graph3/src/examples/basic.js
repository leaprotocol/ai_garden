// @flow
import { NodeFactory } from '../nodes/NodeFactory';

async function runExample() {
  // Create a source node that reads a text file
  const sourceNode = NodeFactory.createNode('source', {
    path: './test.txt',
    chunkSize: 100
  });

  // Create a simple processor that counts words
  const processorNode = NodeFactory.createNode('processor', {
    process: async (input) => {
      const words = input.content.split(/\s+/).length;
      return {
        type: 'stats',
        content: { words },
        metadata: input.metadata
      };
    }
  });

  // Create a sink node that logs output
  const sinkNode = NodeFactory.createNode('sink', {
    process: async (input) => {
      console.log('Chunk stats:', input);
    }
  });

  // Connect nodes via events
  sourceNode.on('processed', async (data) => {
    const result = await processorNode.process(data);
    await sinkNode.process(result);
  });

  // Run the flow
  while (true) {
    await sourceNode.run();
    if (sourceNode.cursorState.position * sourceNode.chunkSize >= stats.size) {
      break;
    }
  }
}

runExample().catch(console.error); 