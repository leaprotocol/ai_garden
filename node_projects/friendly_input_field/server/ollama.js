import { Ollama } from 'ollama';

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';

export function createOllamaClient() {
    const ollama = new Ollama({
        host: OLLAMA_HOST
    });

    console.log(`Ollama client initialized with host: ${OLLAMA_HOST}`);
    return ollama;
}

// Add error handling and retry logic
process.on('unhandledRejection', (error) => {
    console.error('Unhandled Ollama client error:', error);
}); 