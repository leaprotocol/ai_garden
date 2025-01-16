# AI Garden Tools

A comprehensive toolkit for AI-powered automation, testing, and development. This package integrates web automation (Playwright), container management (Docker), and LLM interactions into a unified workflow.

## Features

- **Web Automation**: Browser control and testing using Playwright
- **Docker Management**: Container creation, monitoring and interaction
- **LLM Integration**: Structured interaction with language models
- **Scenario Runner**: Automated execution of complex workflows
- **Extensive Logging**: Detailed activity tracking and debugging support

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai_garden.git

# Navigate to tools directory
cd ai_garden/tools

# Install dependencies
npm install
```

## Usage

### Basic Commands

```bash
# Build the project
npm run build

# Run tests
npm test

# Start the application
npm start
```

### Example Scenario: Android App Development

```typescript
const scenarioRunner = new ScenarioRunner(playwrightModule, dockerModule, llmModule);
await scenarioRunner.runScenario('android-app-dev');
```

## Modules

### Playwright Module
- Browser automation and testing
- Screenshot capture
- Page interaction and navigation
- Form filling and submission

### Docker Module
- Container lifecycle management
- Log streaming and monitoring
- Command execution
- Volume and network management

### LLM Module
- Text generation
- Context management
- Tool calling capabilities
- Response streaming

## Scenarios

Pre-built automation scenarios for common tasks:

1. Web Application Testing
2. Android App Development
3. Content Generation
4. API Testing
5. Documentation Generation

## Development

### Adding New Scenarios

1. Create a new scenario definition in `src/scenarios/`
2. Define steps and actions
3. Register in ScenarioRunner
4. Test and validate

### Module Extension

Each module can be extended with new capabilities:

```typescript
class CustomPlaywrightModule extends PlaywrightModule {
  async customAction() {
    // Implementation
  }
}
```

## Testing

```bash
# Run all tests
npm test

# Watch mode
npm run test:watch

# Coverage report
npm run test:coverage
```

## Logging

The system uses structured logging with different levels:
- DEBUG: Detailed debugging information
- INFO: General operational information
- WARN: Warning messages
- ERROR: Error conditions

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT

## Project Status

Active development - Contributions welcome!

## Future Enhancements

1. Real-time monitoring dashboard
2. Extended LLM capabilities
3. More pre-built scenarios
4. Enhanced error handling
5. Performance optimizations

## Support

- GitHub Issues: Report bugs and feature requests
- Documentation: Comprehensive guides and API reference
- Community: Join our Discord for discussions

## Requirements

- Node.js 18+
- Docker
- Chrome/Chromium (for Playwright)
