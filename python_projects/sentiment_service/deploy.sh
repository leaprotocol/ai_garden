#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y python3-pip python3-venv git

# Create project directory
mkdir -p /root/sentiment_service
cd /root/sentiment_service

# Copy project files (run this locally before running this script)
# scp -P <port> -r /path/to/sentiment_service/* root@<ip>:/root/sentiment_service/

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio

# Install other dependencies
pip install transformers websockets rich

# Run the service
python src/analyzer.py 