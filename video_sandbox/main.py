import torch
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import os

app = FastAPI()

# Load model
model_path = 'openbmb/MiniCPM-V-2_6'
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()

MAX_NUM_FRAMES = 64

def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def encode_image(image):
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    max_size = 448 * 16
    if max(image.size) > max_size:
        w, h = image.size
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    return image

def encode_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    video_frames = vr.get_batch(frame_idx).asnumpy()
    video_frames = [Image.fromarray(frame.astype('uint8')) for frame in video_frames]
    return [encode_image(frame) for frame in video_frames]

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v'}:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Save the file to a temporary location
    file_location = f"/tmp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Process video frames
    video_frames = encode_video(file_location)

    # Perform inference on the frames
    input_text = "Describe the content of the video."  # This could be customized or dynamic
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Example: Pass the video frames to the model (model integration can vary based on use case)
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove the temporary file
    os.remove(file_location)

    return {"message": "Video processed successfully", "frames_processed": len(video_frames), "result": outputs[0].tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
