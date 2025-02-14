#!/bin/bash

set -euo pipefail
trap 'echo "Error on line $LINENO"' ERR

# Configuration
VAST_SSH_PORT=45314
VAST_IP="184.144.229.106"
LOCAL_PORT=8080
REMOTE_PORT=8080
REMOTE_DIR="/root/sentiment_service"
LOCAL_STATUS_FILE=".vastai_deploy_status"

# Colors for pretty output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
info() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"; }

# Update local status
update_status() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOCAL_STATUS_FILE"
    log "$1"
}

# Setup Poetry and install dependencies
setup_poetry() {
    info "Setting up Poetry environment"
    
    # Check if Poetry is installed
    if ! command -v poetry &> /dev/null; then
        info "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
    fi
    
    # Install dependencies
    info "Installing project dependencies with Poetry"
    poetry install
}

# Check SSH connection
check_ssh() {
    info "Checking SSH connection to $VAST_IP:$VAST_SSH_PORT"
    if ! ssh -q -p "$VAST_SSH_PORT" "root@$VAST_IP" exit; then
        error "Cannot connect to Vast.ai instance"
        error "Please check your SSH connection and try again"
        exit 1
    fi
}

# Check if port is available locally
check_port() {
    if lsof -Pi :"$LOCAL_PORT" -sTCP:LISTEN -t >/dev/null ; then
        error "Port $LOCAL_PORT is already in use"
        error "Please free up the port or specify a different one"
        exit 1
    fi
}

# Kill existing port forwards
cleanup_port_forwards() {
    info "Cleaning up existing port forwards"
    pkill -f "ssh.*$VAST_SSH_PORT.*:$LOCAL_PORT" || true
}

# Copy files to remote
copy_files() {
    info "Copying files to Vast.ai instance"
    
    # Ensure source directory exists
    if [ ! -d "src" ]; then
        error "src directory not found"
        exit 1
    fi
    
    # Create remote directory structure
    ssh -p "$VAST_SSH_PORT" "root@$VAST_IP" "mkdir -p $REMOTE_DIR/src"
    
    # Copy files
    scp -P "$VAST_SSH_PORT" \
        src/analyzer.py \
        fabfile.py \
        pyproject.toml \
        poetry.lock \
        "root@$VAST_IP:$REMOTE_DIR/" || {
        error "Failed to copy files"
        exit 1
    }
    
    # Move files to correct locations
    ssh -p "$VAST_SSH_PORT" "root@$VAST_IP" "
        mv $REMOTE_DIR/analyzer.py $REMOTE_DIR/src/
    " || {
        error "Failed to setup files on remote"
        exit 1
    }
}

# Run Fabric deployment
run_fabric_deploy() {
    info "Running Fabric deployment"
    poetry run fab -H "root@$VAST_IP:$VAST_SSH_PORT" deploy
}

# Setup port forwarding
setup_port_forward() {
    update_status "Setting up port forwarding"
    cleanup_port_forwards
    
    ssh -f -N -L "$LOCAL_PORT:localhost:$REMOTE_PORT" -p "$VAST_SSH_PORT" "root@$VAST_IP" || {
        error "Failed to setup port forwarding"
        exit 1
    }
    
    # Verify port forward
    sleep 2
    if ! lsof -Pi :"$LOCAL_PORT" -sTCP:LISTEN -t >/dev/null ; then
        error "Port forwarding failed to start"
        exit 1
    fi
}

# Monitor deployment
monitor_deployment() {
    info "Monitoring remote deployment status"
    
    # Check if service is running
    for i in {1..30}; do
        if ssh -p "$VAST_SSH_PORT" "root@$VAST_IP" \
            "test -f $REMOTE_DIR/service.log && grep -q 'Server running at' $REMOTE_DIR/service.log"; then
            info "Service is running!"
            return 0
        fi
        sleep 1
    done
    
    error "Service failed to start within 30 seconds"
    error "Check remote logs: ssh -p $VAST_SSH_PORT root@$VAST_IP 'cat $REMOTE_DIR/service.log'"
    exit 1
}

main() {
    # Initialize status file
    echo "Starting deployment at $(date)" > "$LOCAL_STATUS_FILE"
    
    # Pre-flight checks
    setup_poetry
    check_ssh
    check_port
    
    # Main deployment steps
    copy_files
    run_fabric_deploy
    setup_port_forward
    monitor_deployment
    
    # Final status
    update_status "Deployment completed successfully"
    
    info "The service is now running on the Vast.ai instance"
    info "Local access: ws://localhost:$LOCAL_PORT"
    info "To monitor the service:"
    info "1. SSH into the instance: ssh -p $VAST_SSH_PORT root@$VAST_IP"
    info "2. Attach to tmux: tmux attach -t sentiment_service"
    info "3. To detach from tmux: Ctrl+B, then D"
    info "4. View deployment log: cat $LOCAL_STATUS_FILE"
    info "5. View remote logs: ssh -p $VAST_SSH_PORT root@$VAST_IP 'tail -f $REMOTE_DIR/service.log'"
}

# Run main function
main "$@" 