import chalk from 'chalk';

export const logger = {
  debug: (...args: any[]) => console.debug(chalk.gray(...args)),
  info: (...args: any[]) => console.info(chalk.blue(...args)),
  warn: (...args: any[]) => console.warn(chalk.yellow(...args)),
  error: (...args: any[]) => console.error(chalk.red(...args)),
};
