#generated 

# MiniCPM Video Processing API

## Description

A FastAPI application that processes video files using the MiniCPM-V-2_6 model.

## Features

- Upload and process video files.
- Extract and handle video frames.
- Perform inference using the MiniCPM-V-2_6 model.

## Setup

### Using pip

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/minicpm-video-api.git
    cd minicpm-video-api
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**

    ```bash
    python main.py
    ```

### Using Docker

1. **Build the Docker image:**

    ```bash
    docker build -t minicpm-video-api .
    ```

2. **Run the Docker container:**

    ```bash
    docker run --gpus all -p 8000:8000 minicpm-video-api
    ```

## API Endpoint

### POST `/process-video/`

- **Description:** Upload a video file for processing.
- **Parameters:**
    - `file`: The video file to upload.
- **Response:**
    - `message`: Success message.
    - `frames_processed`: Number of frames processed.
    - `result`: Model output.

### Example using `curl`:

```bash
curl -X POST "http://localhost:8000/process-video/" -F "file=@your_video.mp4"
