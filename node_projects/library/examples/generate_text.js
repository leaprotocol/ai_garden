import { generate } from '../ollama/ollama_client.js';

async function main() {
  const modelName = 'smollm'; // Replace with the model you want to use
  const promptText = 'Tell me a joke.';

  try {
    const response = await generate(modelName, promptText);
    console.log('Generated Text:', response.response);
  } catch (error) {
    console.error('Failed to generate text:', error);
  }
}

main(); 