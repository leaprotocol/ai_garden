import Docker from 'dockerode';
import { logger } from '../../utils/logger';

export class DockerModule {
  private docker: Docker;

  constructor() {
    this.docker = new Docker();
  }

  async listContainers() {
    logger.info('Listing Docker containers...');
    const containers = await this.docker.listContainers();
    logger.info('Docker containers:', containers);
    return containers;
  }

  async runContainer(image: string, command?: string[]) {
    logger.info(`Running Docker container with image ${image}...`);
    const container = await this.docker.createContainer({
      Image: image,
      Cmd: command,
      Tty: true, // Add Tty option to allocate a pseudo-TTY
      AttachStdout: true, // Attach stdout
      AttachStderr: true, // Attach stderr
    });

    await container.start({});

    container.logs(
      {
        follow: true,
        stdout: true,
        stderr: true,
      },
      (err, stream) => {
        if (err) {
          logger.error('Error streaming logs:', err);
          return;
        }

        if (!stream) {
          logger.error('No log stream available.');
          return;
        }

        stream.on('data', (chunk) => {
          // Process the log data (chunk is a Buffer)
          logger.info('Container log:', chunk.toString('utf8'));
        });

        stream.on('end', () => {
          logger.info('Container log stream ended.');
        });
      },
    );

    logger.info(`Docker container started.`);
    return container;
  }

  // Add more Docker methods as needed...
}
