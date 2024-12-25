import { AbstractNode } from './AbstractNode';
import { Event } from '../types/events';

interface OutputHistory {
  timestamp: number;
  event: Event;
}

export class OutputNode extends AbstractNode {
  private history: OutputHistory[] = [];
  private maxHistory: number = 100;
  private active: boolean = true;

  constructor(id: string) {
    super(id, 'output');
  }

  protected processEvent(event: Event): void {
    if (!this.active) return;

    // Add to history
    this.history.push({
      timestamp: Date.now(),
      event
    });

    // Trim history if needed
    if (this.history.length > this.maxHistory) {
      this.history = this.history.slice(-this.maxHistory);
    }

    // Format and log the event
    this.logEvent(event);
  }

  private logEvent(event: Event): void {
    let message = '';

    switch (event.meta?.type) {
      case 'text':
        message = event.done ? 
          `Complete: ${event.data}` :
          `Chunk [${event.meta.seq}]: ${event.data}`;
        break;

      case 'error':
        message = `Error: ${event.data.error}`;
        if (event.data.details) {
          message += `\nDetails: ${JSON.stringify(event.data.details)}`;
        }
        break;

      case 'state':
        message = `State: ${event.data.state}`;
        if (event.data.details) {
          message += ` (${JSON.stringify(event.data.details)})`;
        }
        break;

      default:
        message = `Data: ${JSON.stringify(event.data)}`;
    }

    this.logDebug(message);
  }

  public clearHistory(): void {
    this.history = [];
    this.emitState('cleared');
  }

  public setActive(active: boolean): void {
    this.active = active;
    this.emitState('active', { active });
  }

  public getHistory(): OutputHistory[] {
    return [...this.history];
  }

  public setMaxHistory(max: number): void {
    this.maxHistory = max;
    if (this.history.length > max) {
      this.history = this.history.slice(-max);
    }
    this.emitState('config', { maxHistory: max });
  }
} 