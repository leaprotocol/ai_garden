#!/bin/bash

set -euo pipefail  # Strict mode
trap 'echo "Error on line $LINENO"' ERR

# Status file to track deployment progress
STATUS_FILE="/root/sentiment_service/.deploy_status"
LOCK_FILE="/root/sentiment_service/.deploy_lock"

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
info() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"; }

# Update deployment status
update_status() {
    echo "$1" > "$STATUS_FILE"
    log "$1"
}

# Check if another deployment is running
check_lock() {
    if [ -f "$LOCK_FILE" ]; then
        pid=$(cat "$LOCK_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            error "Another deployment is running (PID: $pid)"
            exit 1
        fi
        rm "$LOCK_FILE"
    fi
    echo $$ > "$LOCK_FILE"
}

# Cleanup function
cleanup() {
    rm -f "$LOCK_FILE"
    if [ $? -eq 0 ]; then
        update_status "Deployment completed successfully"
    else
        update_status "Deployment failed"
    fi
}

check_gpu() {
    update_status "Checking GPU availability"
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. Is this a GPU instance?"
        exit 1
    fi
    
    nvidia-smi
    gpu_count=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    info "Found $gpu_count GPU(s)"
    
    # Log GPU details
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv,noheader > "$STATUS_FILE.gpu"
}

install_system_deps() {
    update_status "Installing system dependencies"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq \
        python3-pip \
        htop \
        tmux \
        nvtop \
        2>&1 | tee -a "$STATUS_FILE.apt"
}

setup_python_deps() {
    update_status "Setting up Python dependencies"
    
    log "Upgrading pip"
    pip install --upgrade pip -q

    log "Installing Python packages"
    pip install --no-cache-dir \
        transformers \
        websockets \
        rich \
        nvitop \
        2>&1 | tee -a "$STATUS_FILE.pip"

    log "Verifying PyTorch CUDA"
    python3 -c '
import torch
import sys
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
sys.exit(0 if torch.cuda.is_available() else 1)
' 2>&1 | tee -a "$STATUS_FILE.torch"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        error "PyTorch CUDA verification failed"
        exit 1
    fi
}

setup_monitoring() {
    update_status "Setting up monitoring"
    tmux kill-session -t sentiment_service 2>/dev/null || true
    
    # Create new tmux session with monitoring layout
    TMUX= tmux new-session -d -s sentiment_service
    
    # Configure tmux
    tmux set-option -t sentiment_service mouse on
    tmux set-option -t sentiment_service history-limit 50000
    
    # Create monitoring layout
    tmux split-window -h
    tmux split-window -v
    tmux select-pane -t 0
    tmux split-window -v
    
    # Setup monitoring tools
    tmux select-pane -t 0
    tmux send-keys "htop" C-m
    
    tmux select-pane -t 1
    tmux send-keys "watch -n1 'nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=table'" C-m
    
    tmux select-pane -t 2
    tmux send-keys "nvitop" C-m
    
    tmux select-pane -t 3
    
    info "Monitoring setup complete in tmux"
}

start_service() {
    update_status "Starting sentiment analysis service"
    tmux select-pane -t 3
    tmux send-keys "cd /root/sentiment_service && python src/analyzer.py 2>&1 | tee service.log" C-m
    
    # Wait for service to start
    for i in {1..30}; do
        if grep -q "Server running at" service.log 2>/dev/null; then
            info "Service started successfully"
            return 0
        fi
        sleep 1
    done
    error "Service failed to start within 30 seconds"
    exit 1
}

main() {
    mkdir -p /root/sentiment_service
    cd /root/sentiment_service || exit 1
    
    # Setup cleanup trap
    trap cleanup EXIT
    
    # Check for existing deployment
    check_lock
    
    update_status "Starting deployment"
    
    check_gpu
    install_system_deps
    setup_python_deps
    setup_monitoring
    start_service
    
    info "Deployment successful!"
    info "To view the service and monitoring:"
    info "1. Run: tmux attach -t sentiment_service"
    info "2. To detach from tmux: press Ctrl+B, then D"
    info "3. Service is running on port 8080"
    info "4. View deployment status: cat $STATUS_FILE"
    info "5. View service logs: tail -f service.log"
}

# Run main function
main "$@" 