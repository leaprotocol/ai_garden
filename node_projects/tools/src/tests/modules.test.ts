import { PlaywrightModule } from '../modules/playwright/playwright';
import { DockerModule } from '../modules/docker/docker';
import { LLMModule } from '../modules/llm/llm';

describe('PlaywrightModule', () => {
  it('should initialize', async () => {
    const playwrightModule = new PlaywrightModule();
    await playwrightModule.init();
    expect(playwrightModule).toBeTruthy();
    await playwrightModule.close();
  });

  // Add more tests for PlaywrightModule methods...
});

describe('DockerModule', () => {
  it('should list containers', async () => {
    const dockerModule = new DockerModule();
    const containers = await dockerModule.listContainers();
    expect(containers).toBeTruthy();
  });

  // Add more tests for DockerModule methods...
});

describe('LLMModule', () => {
  it('should generate text', async () => {
    const llmModule = new LLMModule();
    const text = await llmModule.generateText('Test prompt');
    expect(text).toBeTruthy();
  });

  // Add more tests for LLMModule methods...
});
