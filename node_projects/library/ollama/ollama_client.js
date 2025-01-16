import { Ollama } from 'ollama';
import { EventEmitter } from 'events';

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';

export async function getAvailableModels() {
  const ollamaClient = new Ollama({ host: OLLAMA_HOST });
  try {
    const response = await ollamaClient.list();
    return response.models.map(model => ({
      name: model.name,
      size: model.size,
      modified: model.modified_at,
      digest: model.digest
    }));
  } catch (error) {
    console.error('Error fetching Ollama models:', error);
    throw error;
  }
}

export class OllamaStreamHandler extends EventEmitter {
  constructor(stream, client) {
    super();
    this.stream = stream;
    this.client = client; // Instance of Ollama client
    this.tokenCount = 0;
    this.currentMessage = '';
    this.setupListeners();
  }

  async setupListeners() {
    try {
      for await (const chunk of this.stream) {
        const data = chunk;
        if (data.response) {
          this.currentMessage += data.response;
          this.tokenCount += data.response.split(' ').length; // Simple token counting
          this.emit('data', {data, wholeMessage: this.currentMessage, tokenCount: this.tokenCount});
        }
        if (data.done) {
          this.emit('end');
        }
        if (data.error) {
          this.emit('error', new Error(data.error));
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        this.emit('abort');
      } else {
        this.emit('error', error);
      }
    }
  }


  getCurrentMessage() {
    return this.currentMessage;
  }

  abort() {
    if (this.client) {
      this.client.abort();
    }
  }
}

export async function generateStream({model, prompt, options = {}, modelOptions = {}}) {
  try {
    const ollamaClient = new Ollama({ 
        host: OLLAMA_HOST, 
    }); // New client per stream

    const stream = await ollamaClient.generate({
        model,
        prompt: prompt,
        stream: true,
        raw: false,
        ...options,
        options: {
            ...modelOptions,
            //stop: []//['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '/INST'],
        },
        //images:["iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII="],
        //context: 
        
    }); 
    return new OllamaStreamHandler(stream, ollamaClient);
  } catch (error) {
    console.error('Error generating Ollama stream:', error);
    throw error;
  }
}
