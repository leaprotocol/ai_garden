from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
from smol_hello.main import load_model, generate_text, generate_text_stream  # Import your existing functions
from typing import AsyncIterator
import json
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Smol Hello API", description="Simple API to generate text using SmolLM2")

# Load model and tokenizer when the app starts
model, tokenizer = load_model()

# After creating the FastAPI app, add:
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class TextRequest(BaseModel):
    text_prompt: str

class TextResponse(BaseModel):
    generated_text: str

@app.websocket("/ws/generate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive and parse the prompt
            data = await websocket.receive_text()
            request = json.loads(data)
            prompt = request.get("text_prompt", "")
            
            # Stream the generated text
            async for token in generate_text_stream(model, tokenizer, prompt):
                await websocket.send_text(json.dumps({"token": token}))
                
            # Send end message
            await websocket.send_text(json.dumps({"done": True}))
            
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        await websocket.close()

@app.post("/generate/", response_model=TextResponse)
async def generate(request: TextRequest):
    try:
        response_text = generate_text(model, tokenizer, request.text_prompt)
        return {"generated_text": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

def run_api():
    """Runs the FastAPI application using Uvicorn."""
    import uvicorn
    uvicorn.run("smol_hello.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run_api() # You can run the API directly for testing 