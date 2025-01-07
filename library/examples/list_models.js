import { getAvailableModels } from '../ollama/ollama_client.js';

async function main() {
  try {
    const models = await getAvailableModels();
    console.log('Available Ollama Models:');
    console.log(JSON.stringify(models, null, 2));
  } catch (error) {
    console.error('Failed to list models:', error);
  }
}

main(); 