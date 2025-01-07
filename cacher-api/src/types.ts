export interface ProcessOptions {
  maxLength?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  numBeams?: number;
  useCache?: boolean;
}

export interface ContinuationOptions {
  cachedState: {
    past_key_values: any;
    input_ids: any;
    attention_mask: any;
  };
  suffix: string;
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
}

export interface CacheResult {
  generatedText: string;
  cachedState: {
    past_key_values: any;
    input_ids: any;
    attention_mask: any;
  };
  tokenProbabilities?: Array<TokenProbability>;
}

export interface TokenProbability {
  token: string;
  probability: number;
  entropy?: number;
}

export interface LogConfig {
  level: 'debug' | 'info' | 'warn' | 'error';
  enableConsole?: boolean;
  enableFile?: boolean;
  filePath?: string;
}

export interface ModelConfig {
  modelName: string;
  device?: 'cpu' | 'gpu';
  useQuantization?: boolean;
  maxLength?: number;
  defaultTemperature?: number;
}
