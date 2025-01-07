import EventEmitter from 'events';
import TypedEmitter from 'typed-emitter';
import { LogConfig } from '../types';

type LogEvents = {
  debug: (message: string, ...args: any[]) => void;
  info: (message: string, ...args: any[]) => void;
  warn: (message: string, ...args: any[]) => void;
  error: (message: string, ...args: any[]) => void;
};

export class Logger extends (EventEmitter as new () => TypedEmitter<LogEvents>) {
  private config: LogConfig;

  constructor(config?: Partial<LogConfig>) {
    super();
    this.config = {
      level: 'info',
      enableConsole: true,
      enableFile: false,
      filePath: './logs/cacher.log',
      ...config
    };

    this.setupListeners();
  }

  private setupListeners() {
    if (this.config.enableConsole) {
      this.on('debug', (message, ...args) => {
        if (this.shouldLog('debug')) console.debug(`[DEBUG] ${message}`, ...args);
      });

      this.on('info', (message, ...args) => {
        if (this.shouldLog('info')) console.info(`[INFO] ${message}`, ...args);
      });

      this.on('warn', (message, ...args) => {
        if (this.shouldLog('warn')) console.warn(`[WARN] ${message}`, ...args);
      });

      this.on('error', (message, ...args) => {
        if (this.shouldLog('error')) console.error(`[ERROR] ${message}`, ...args);
      });
    }
  }

  private shouldLog(level: string): boolean {
    const levels = ['debug', 'info', 'warn', 'error'];
    return levels.indexOf(level) >= levels.indexOf(this.config.level);
  }

  debug(message: string, ...args: any[]) {
    this.emit('debug', message, ...args);
  }

  info(message: string, ...args: any[]) {
    this.emit('info', message, ...args);
  }

  warn(message: string, ...args: any[]) {
    this.emit('warn', message, ...args);
  }

  error(message: string, ...args: any[]) {
    this.emit('error', message, ...args);
  }
}

