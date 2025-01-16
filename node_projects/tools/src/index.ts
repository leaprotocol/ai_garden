import { logger } from './utils/logger';
import { PlaywrightModule } from './modules/playwright';
import { DockerModule } from './modules/docker';
import { LLMModule } from './modules/llm';
import { ScenarioRunner } from './scenarios/scenarioRunner';

async function main() {
  logger.info('Starting AI Garden Tools...');

  // Initialize modules
  const playwrightModule = new PlaywrightModule();
  const dockerModule = new DockerModule();
  const llmModule = new LLMModule();

  // Example usage (you can replace this with your desired logic)
  const scenarioRunner = new ScenarioRunner(playwrightModule, dockerModule, llmModule);
  await scenarioRunner.runScenario('android-app-dev');

  logger.info('AI Garden Tools finished.');
}

main().catch((error) => {
  logger.error('An error occurred:', error);
  process.exit(1);
});
