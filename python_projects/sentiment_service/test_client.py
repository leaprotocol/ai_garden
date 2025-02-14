import asyncio
import websockets
import json

async def test_sentiment():
    uri = "ws://localhost:8080"
    async with websockets.connect(uri) as websocket:
        # Test single text
        request = {"text": "Bitcoin is showing strong momentum!"}
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        print(f"Single text analysis: {response}")
        
        # Test batch of texts
        request = {
            "texts": [
                "Market sentiment is very negative",
                "New partnership announced with major tech company"
            ]
        }
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        print(f"Batch analysis: {response}")

if __name__ == "__main__":
    asyncio.run(test_sentiment()) 