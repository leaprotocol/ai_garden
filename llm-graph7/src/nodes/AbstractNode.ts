import { Event, createEvent, createErrorEvent, createStateEvent } from '../types/events';
import { EventEmitter } from 'events';

export abstract class AbstractNode extends EventEmitter {
  protected id: string;
  protected type: string;
  protected debug: boolean = false;
  protected connections: Set<AbstractNode> = new Set();

  constructor(id: string, type: string) {
    super();
    this.id = id;
    this.type = type;
  }

  // Handle incoming events
  public handleEvent(event: Event): void {
    try {
      if (this.debug) {
        this.logDebug(`Received event: ${JSON.stringify(event)}`);
      }
      
      // Let derived classes handle the event
      this.processEvent(event);
    } catch (error) {
      this.emitError(error instanceof Error ? error : new Error(String(error)));
    }
  }

  // Abstract method for processing events - must be implemented by derived classes
  protected abstract processEvent(event: Event): void;

  // Connect to another node
  public connect(node: AbstractNode): void {
    this.connections.add(node);
    if (this.debug) {
      this.logDebug(`Connected to node: ${node.id}`);
    }
  }

  // Disconnect from a node
  public disconnect(node: AbstractNode): void {
    this.connections.delete(node);
    if (this.debug) {
      this.logDebug(`Disconnected from node: ${node.id}`);
    }
  }

  // Emit an event to all connected nodes
  protected emit(data: any, done: boolean = true, meta: Partial<Event['meta']> = {}): void {
    const event = createEvent(this.id, data, done, meta);
    this.broadcastEvent(event);
  }

  // Broadcast event to all connected nodes
  protected broadcastEvent(event: Event): void {
    if (this.debug) {
      this.logDebug(`Broadcasting event: ${JSON.stringify(event)}`);
    }
    
    for (const node of this.connections) {
      node.handleEvent(event);
    }
  }

  // Emit an error event
  protected emitError(error: Error): void {
    const errorEvent = createErrorEvent(this.id, error);
    this.broadcastEvent(errorEvent);
  }

  // Emit a state change event
  protected emitState(state: string, details?: any): void {
    const stateEvent = createStateEvent(this.id, state, details);
    this.broadcastEvent(stateEvent);
  }

  // Toggle debug mode
  public setDebug(enabled: boolean): void {
    this.debug = enabled;
    this.emitState('debug', { enabled });
  }

  // Debug logging
  protected logDebug(message: string): void {
    if (this.debug) {
      console.log(`[${this.type}:${this.id}] ${message}`);
    }
  }

  // Cleanup resources
  public destroy(): void {
    this.connections.clear();
    this.removeAllListeners();
  }
} 