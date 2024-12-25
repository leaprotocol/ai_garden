function log(level, message, ...args) {
  const timestamp = new Date().toLocaleTimeString();
  console[level](`[${timestamp}] ${message}`, ...args);
}

export const logger = {
  info: (message, ...args) => log('info', message, ...args),
  warn: (message, ...args) => log('warn', message, ...args),
  error: (message, ...args) => log('error', message, ...args),
};

export function displayError(message) {
  // You can customize how errors are displayed to the user
  alert(`Error: ${message}`);
} 