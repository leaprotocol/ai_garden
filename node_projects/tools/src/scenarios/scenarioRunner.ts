import { logger } from '../utils/logger';
import { PlaywrightModule } from '../modules/playwright';
import { DockerModule } from '../modules/docker';
import { LLMModule } from '../modules/llm';
import { Scenario, Step } from '../types';

export class ScenarioRunner {
  private playwrightModule: PlaywrightModule;
  private dockerModule: DockerModule;
  private llmModule: LLMModule;

  constructor(
    playwrightModule: PlaywrightModule,
    dockerModule: DockerModule,
    llmModule: LLMModule,
  ) {
    this.playwrightModule = playwrightModule;
    this.dockerModule = dockerModule;
    this.llmModule = llmModule;
  }

  async runScenario(scenarioName: string) {
    logger.info(`Running scenario: ${scenarioName}`);

    // Load scenario definition (you might want to load this from a file)
    const scenario = this.getScenarioDefinition(scenarioName);

    if (!scenario) {
      logger.error(`Scenario not found: ${scenarioName}`);
      return;
    }

    // Initialize modules
    await this.playwrightModule.init();

    // Execute scenario steps
    for (const step of scenario.steps) {
      logger.info(`Executing step: ${step.name}`);
      try {
        await step.action({
          playwright: this.playwrightModule,
          docker: this.dockerModule,
          llm: this.llmModule,
        });
      } catch (error) {
        logger.error(`Error executing step ${step.name}:`, error);
        break; // Stop execution on error (you might want to handle this differently)
      }
    }
    logger.info(`Scenario finished: ${scenarioName}`);

    // Close modules
    await this.playwrightModule.close();
  }

  private getScenarioDefinition(scenarioName: string): Scenario | null {
    // Replace this with your actual scenario loading logic
    const scenarios: Record<string, Scenario> = {
      'android-app-dev': {
        name: 'Android App Development',
        description: 'Develop and test a simple Android app',
        steps: [
          {
            name: 'Run Ubuntu container',
            action: async ({ docker }: { docker: DockerModule }) => {
              await docker.runContainer('ubuntu', ['bash']);
            },
          },
          // Add more steps for Android app development...
        ],
      },
      // Define more scenarios here...
    };

    return scenarios[scenarioName] || null;
  }
}
