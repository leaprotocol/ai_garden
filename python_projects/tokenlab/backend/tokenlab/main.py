from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
from .model import TokenAnalyzer
from pathlib import Path
import uvicorn

app = FastAPI()

# Mount static files
frontend_path = Path(__file__).parent.parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Initialize model
analyzer = TokenAnalyzer()

@app.get("/")
async def get_root():
    """Serve the main HTML page."""
    html_file = frontend_path / "index.html"
    with open(html_file) as f:
        return HTMLResponse(f.read())

@app.get("/ascii")
async def get_ascii():
    """Serve the ASCII visualization page."""
    html_file = frontend_path / "ascii.html"
    with open(html_file) as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'analyze':
                # Analyze tokens
                token_data = await analyzer.get_token_probabilities(data['text'])
                await websocket.send_json({
                    'type': 'analysis',
                    'text': data['text'],
                    'tokens': token_data
                })
            elif data.get('type') == 'generate':
                async for token_data in analyzer.generate_stream(data['prompt'], max_length=5):
                    if token_data['type'] == 'initial_tokens':
                        # Send initial tokenization
                        await websocket.send_json({
                            'type': 'analysis',
                            'tokens': token_data['tokens']
                        })
                    else:
                        # Send next generated token
                        await websocket.send_json({
                            'type': 'generation',
                            'token': token_data
                        })
                
                await websocket.send_json({
                    'type': 'generation_complete'
                })
            elif data.get('type') == 'stop_generation':
                # Client requested to stop generation
                continue
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

def run_server():
    """Entry point for the poetry script."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server() 