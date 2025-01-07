import { AutoTokenizer } from '@huggingface/transformers';

export class TokenizerUtils {
  private tokenizer: any;

  constructor(modelName: string) {
    this.initializeTokenizer(modelName);
  }

  private async initializeTokenizer(modelName: string) {
    this.tokenizer = await AutoTokenizer.from_pretrained(modelName);
  }

  async encode(text: string): Promise<number[]> {
    return await this.tokenizer.encode(text);
  }

  async decode(tokens: number[]): Promise<string> {
    return await this.tokenizer.decode(tokens);
  }

  async getVocabSize(): Promise<number> {
    return this.tokenizer.vocab_size;
  }

  async tokenize(text: string): Promise<string[]> {
    return await this.tokenizer.tokenize(text);
  }

  async calculateTokenProbabilities(
    logits: Float32Array,
    tokens: number[]
  ): Promise<Array<{ token: string; probability: number }>> {
    const probabilities = [];
    const softmaxValues = this.softmax(Array.from(logits));

    for (let i = 0; i < tokens.length; i++) {
      const token = await this.decode([tokens[i]]);
      probabilities.push({
        token,
        probability: softmaxValues[tokens[i]]
      });
    }

    return probabilities;
  }

  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expValues = logits.map(l => Math.exp(l - maxLogit));
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    return expValues.map(exp => exp / sumExp);
  }
}

