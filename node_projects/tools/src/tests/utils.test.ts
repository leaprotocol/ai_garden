import { logger } from '../utils/logger';

describe('logger', () => {
  it('should log info messages', () => {
    const spy = jest.spyOn(console, 'info').mockImplementation();
    logger.info('Test info message');
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  // Add more tests for other logger methods...
});
