# TokenLab

Interactive token probability visualization tool using FastAPI and ChartJS. Analyze token probabilities in real-time as you type.

## Features
- Real-time token probability visualization
- Async token probability calculation
- Interactive ChartJS graphs
- WebSocket streaming for live updates

## Structure
```
tokenlab/
├── backend/
│   └── tokenlab/
│       ├── __init__.py
│       ├── main.py          # FastAPI server
│       ├── model.py         # Model handling
│       └── utils.py         # Helper functions
├── frontend/
│   ├── index.html          # Main UI
│   ├── style.css           # Styling
│   └── script.js           # ChartJS and WebSocket logic
├── pyproject.toml          # Poetry configuration
└── README.md              # Project documentation
```

## Installation

```bash
# Install dependencies using poetry
poetry install

# Run the server
poetry run tokenlab-server
```

## Development

```bash
# Run with auto-reload
poetry run uvicorn tokenlab.main:app --reload
``` 