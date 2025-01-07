import { AutoTokenizer, AutoModelForCausalLM } from '@huggingface/transformers';
import { Logger } from './utils/logging.js';

export class StateCacher {
  private model: any;
  private tokenizer: any;
  private logger: Logger;
  private initialized: Promise<void>;

  constructor(options: {
    modelName: string;
    device?: "gpu";
    useQuantization?: boolean;
  }) {
    this.logger = new Logger({ level: 'debug' });
    this.logger.debug('StateCacher constructor called with options:', options);
    this.initialized = this.initializeModel(options);
    this.generateNextToken = this.generateNextToken.bind(this); // Bind the method
  }

  private async initializeModel(options: {
    modelName: string;
    device?: "gpu";
    useQuantization?: boolean;
  }) {
    try {
      this.logger.info(`Loading tokenizer for ${options.modelName}`);
      this.tokenizer = await AutoTokenizer.from_pretrained(options.modelName);
      //this.logger.debug('Tokenizer loaded:', this.tokenizer);

      this.logger.info(`Loading model ${options.modelName} with options:`, {
        device: options.device
      });
      this.model = await AutoModelForCausalLM.from_pretrained(options.modelName, {
        device: options.device
      });
      this.logger.info(`Model ${options.modelName} loaded successfully`);
      //this.logger.debug('Model details:', this.model.toJSON());
    } catch (error) {
      this.logger.error('Error loading model or tokenizer:', error);
      throw error;
    }
  }

  async generateNextToken(text: string, numTokens: number = 5): Promise<string[]> {
    await this.initialized;
    this.logger.debug(`generateNextToken called with text: "${text}"`);

    if (!this.tokenizer || !this.model) {
      const error = new Error('Model or tokenizer not initialized');
      this.logger.error('Error:', error);
      throw error;
    }

    try {
      this.logger.debug(`Encoding text: "${text}"`);
      const inputs = await this.tokenizer(text, { 
        //return_tensors: 'pt',
        //padding: true
      });
      this.logger.debug('Encoded inputs:', inputs);

      // Configure generation parameters
      const generationConfig = {
        max_new_tokens: 2,
        do_sample: true,
        num_return_sequences: 5, // Generate multiple sequences
        //pad_token_id: this.tokenizer.pad_token_id,
        //eos_token_id: this.tokenizer.eos_token_id
        //output_scores: true,
        return_dict_in_generate: true,
      };

      this.logger.debug('Generation config:', generationConfig);

      // Generate sequences
      const outputs = await this.model.generate({...inputs, ...generationConfig});
      this.logger.debug('Raw generation outputs:', Object.keys(outputs), outputs.possibleTokens);

      // Extract the generated token IDs from the sequences tensor
      const sequences = outputs.sequences;
      const sequencesArray = Array.from(sequences.ort_tensor.cpuData);
      const sequenceLength = sequences.ort_tensor.dims[1];

      // Get the last token from each sequence
      const lastTokens = [];
      for (let i = sequenceLength - 1; i < sequencesArray.length; i += sequenceLength) {
        lastTokens.push(sequencesArray[i]);
      }

      // Count the occurrences of each token
      const tokenCounts = new Map<bigint, number>();
      for (const token of lastTokens) {
        tokenCounts.set(token, (tokenCounts.get(token) || 0) + 1);
      }

      // Calculate probabilities based on counts
      const tokenProbabilities = Array.from(tokenCounts.entries()).map(([token, count]) => ({
        token,
        probability: count / numTokens
      }));

      // Sort by probability
      tokenProbabilities.sort((a, b) => b.probability - a.probability);

      // Decode tokens and format output
      const possibleTokens = await Promise.all(
        tokenProbabilities.map(async ({ token, probability }) => {
          const decoded = await this.tokenizer.decode([Number(token)], {
            skip_special_tokens: true,
            clean_up_tokenization_spaces: true
          });
          this.logger.debug(`Token: "${decoded}", Probability: ${(probability * 100).toFixed(1)}%`);
          return `${decoded} (${(probability * 100).toFixed(1)}%)`;
        })
      );

      this.logger.info(`Generated possible tokens: ${possibleTokens.join(', ')}`);
      return possibleTokens;

    } catch (error) {
      this.logger.error('Error in generateNextToken:', error);
      throw error;
    }
  }

  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expScores = logits.map(l => Math.exp(l - maxLogit));
    const expSum = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(e => e / expSum);
  }

  private getTopK(array: number[], k: number): number[] {
    return array
      .map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value)
      .slice(0, k)
      .map(item => item.index);
  }
}