import { AbstractNode } from './AbstractNode';
import { Event, createStreamEvent } from '../types/events';

interface LLMConfig {
  model: string;
  prompt?: string;
}

export class LLMNode extends AbstractNode {
  private config: LLMConfig;
  private streamSequence: number = 0;
  private currentRequestId: string | null = null;

  constructor(id: string, config: LLMConfig) {
    super(id, 'llm');
    this.config = config;
  }

  protected async processEvent(event: Event): Promise<void> {
    // Only process text events when not already processing
    if (this.currentRequestId || event.meta?.type !== 'text') {
      return;
    }

    try {
      this.currentRequestId = event.id;
      this.streamSequence = 0;
      
      // Start processing state
      this.emitState('processing', { prompt: event.data });

      // Connect to Ollama
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.config.model,
          prompt: event.data,
          stream: true
        })
      });

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Failed to get response reader');
      }

      // Process the stream
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          // Emit final completion event
          this.emit(this.config.prompt, true, {
            type: 'text',
            parent: event.id,
            seq: ++this.streamSequence
          });
          break;
        }

        // Parse and emit chunk
        const chunk = new TextDecoder().decode(value);
        const json = JSON.parse(chunk);
        
        if (json.response) {
          const streamEvent = createStreamEvent(
            this.id,
            json.response,
            ++this.streamSequence,
            false,
            event.id
          );
          this.broadcastEvent(streamEvent);
        }
      }

      this.emitState('complete');

    } catch (error) {
      this.emitError(error instanceof Error ? error : new Error(String(error)));
      this.emitState('error');
    } finally {
      this.currentRequestId = null;
      this.streamSequence = 0;
    }
  }

  public setModel(model: string): void {
    this.config.model = model;
    this.emitState('config', { model });
  }

  public setPrompt(prompt: string): void {
    this.config.prompt = prompt;
    this.emitState('config', { prompt });
  }
} 