import { StateCacher } from './StateCacher.js';
import { Logger } from './utils/logging.js';

const logger = new Logger({ level: 'debug' });

async function runDemo() {
  try {
    // Initialize the cacher
    logger.info("Initializing StateCacher...");
    const cacher = new StateCacher({
      modelName: "HuggingFaceTB/SmolLM2-360M-Instruct",
      device: "cpu"
    });
    logger.debug("StateCacher instance created:", cacher);
    logger.info("StateCacher initialized.");

    // Wait for initialization to complete
    logger.info("Waiting for initialization to complete...");
    await cacher.initialized;
    logger.info("Initialization complete.");

    const initialText = "The quick brown fox";

    // Generate the next token
    logger.info(`Generating first token for: "${initialText}"`);
    const nextToken1 = await cacher.generateNextToken(initialText);
    logger.info(`First generated token: "${nextToken1}"`);

    // Generate the next token again with the same initial text
    logger.info(`Generating second token for: "${initialText}"`);
    const nextToken2 = await cacher.generateNextToken(initialText);
    logger.info(`Second generated token: "${nextToken2}"`);

  } catch (error) {
    logger.error("Demo failed:", error);
  }
}

runDemo();
