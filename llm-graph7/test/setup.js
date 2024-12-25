// Increase timeout for all tests
jest.setTimeout(15000);

// Add custom matchers if needed
expect.extend({
  toBeWithinRange(received, floor, ceiling) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () =>
          `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true,
      };
    } else {
      return {
        message: () =>
          `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false,
      };
    }
  },
});

// Global test setup
beforeAll(() => {
  // Any global setup needed
  console.log('Starting test suite...');
});

// Global test teardown
afterAll(() => {
  // Any global cleanup needed
  console.log('Test suite completed.');
}); 