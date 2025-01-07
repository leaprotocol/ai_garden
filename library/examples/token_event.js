import { generateStream } from '../ollama/ollama_client.js';

async function main() {
  const modelName = 'smollm';
  const promptText = 'Write a short story about a cat.';

  try {
    const streamHandler = await generateStream({model: modelName, prompt: promptText, modelOptions: { num_thread: 1 }});

    streamHandler.on('token', (token) => {
      process.stdout.write(token);
    });

    streamHandler.onTokenCount(20, (currentMessage) => {
      console.log('\nReached 20 tokens! Current message:\n', currentMessage);
    });

    streamHandler.on('tokenCount', (currentCount) => {
      if (currentCount >= 40) {
        console.log('\nReached 40 tokens! Aborting the stream.');
        streamHandler.abort();
      }
    });

    const interval = setInterval(() => {
      if (streamHandler.tokenCount >= 10) {
        console.log('\nInterval check - Current message (after 10 tokens):\n', streamHandler.getCurrentMessage());
      }
    }, 5000); // Check every 5 seconds

    streamHandler.on('abort', () => {
      console.log('\n-- Stream Aborted --');
      clearInterval(interval);
    });

    streamHandler.on('end', () => {
      console.log('\n-- Stream Complete --');
      clearInterval(interval);
    });

    streamHandler.on('error', (error) => {
      console.error('Stream error:', error);
      clearInterval(interval);
    });
  } catch (error) {
    console.error('Failed to generate stream:', error);
  }
}

main();