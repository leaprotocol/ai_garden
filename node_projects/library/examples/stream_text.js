import { generateStream } from '../ollama/ollama_client.js';

async function main() {
  const modelName = 'llama2'; // Replace with the model you want to use
  const promptText = 'Describe the city of the future.';

  try {
    const streamHandler = await generateStream(modelName, promptText);

    streamHandler.on('token', (token) => {
      process.stdout.write(token); // Stream output to console
    });

    streamHandler.on('end', () => {
      console.log('\n-- Stream Complete --');
    });

    streamHandler.on('error', (error) => {
      console.error('Stream error:', error);
    });
  } catch (error) {
    console.error('Failed to stream text:', error);
  }
}

main(); 