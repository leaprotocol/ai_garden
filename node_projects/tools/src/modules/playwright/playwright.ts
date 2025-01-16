import { chromium, Browser, Page } from 'playwright';
import { logger } from '../../utils/logger';

export class PlaywrightModule {
  private browser: Browser | null = null;
  private page: Page | null = null;

  async init() {
    logger.info('Initializing Playwright...');
    this.browser = await chromium.launch();
    this.page = await this.browser.newPage();
    logger.info('Playwright initialized.');
  }

  async goto(url: string) {
    if (!this.page) {
      throw new Error('Playwright not initialized.');
    }
    logger.info(`Navigating to ${url}...`);
    await this.page.goto(url);
    logger.info(`Navigated to ${url}.`);
  }

  async close() {
    logger.info('Closing Playwright...');
    if (this.browser) {
      await this.browser.close();
    }
    logger.info('Playwright closed.');
  }

  // Add more Playwright methods as needed...
}
