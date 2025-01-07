import { StateCacher } from '../src/StateCacher.js';
import { Logger } from '../src/utils/logging.js';

// Mock the transformers library
jest.mock('@huggingface/transformers', () => ({
  AutoTokenizer: {
    from_pretrained: jest.fn().mockResolvedValue({
      encode: jest.fn().mockResolvedValue([1, 2, 3]),
      decode: jest.fn().mockResolvedValue("test"),
      pad_token_id: 0,
      eos_token_id: 2,
      apply_chat_template: jest.fn().mockReturnValue({
        input_ids: [[1, 2, 3]],
        attention_mask: [[1, 1, 1]],
      }),
    })
  },
  AutoModelForCausalLM: {
    from_pretrained: jest.fn().mockResolvedValue({
      generate: jest.fn().mockImplementation((input_ids, options) => {
        if (options && options.past_key_values) {
          // If past_key_values are provided, simulate generating a continuation
          return Promise.resolve([7, 8, 9]);
        } else {
          // Simulate generating initial text
          return Promise.resolve({
            logits: new Float32Array([0.1, 0.2, 0.3]),
            past_key_values: {},
            sequences: [[4, 5, 6]],
          });
        }
      }),
      config: {
        pad_token_id: 0
      }
    })
  },
  Tensor: {
    cat: jest.fn().mockImplementation((tensors, dim) => {
      // Simulate tensor concatenation
      const concatenated = tensors.reduce((acc, tensor) => {
        return acc.concat(tensor.flat());
      }, []);
      return [concatenated];
    }),
  }
}));

describe('StateCacher', () => {
  let cacher: StateCacher;

  beforeEach(() => {
    cacher = new StateCacher({
      modelName: 'test-model',
      device: 'cpu'
    });
  });

  test('initialization', () => {
    expect(cacher).toBeDefined();
  });

  test('processAndCache generates text and returns cached state', async () => {
    const result = await cacher.processAndCache("Test prompt");
    expect(result.generatedText).toBeDefined();
    expect(result.generatedText).toEqual("test");
    expect(result.cachedState).toBeDefined();
  });

  test('generateContinuation uses cached state', async () => {
    const { cachedState } = await cacher.processAndCache("Test prompt");
    const continuation = await cacher.generateContinuation({
      cachedState,
      suffix: "Test continuation",
      maxNewTokens: 3
    });
    expect(continuation).toBeDefined();
    expect(continuation).toEqual("test");
  });
});
