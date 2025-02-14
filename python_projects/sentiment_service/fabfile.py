from fabric import Connection, task
import logging
from rich.logging import RichHandler
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("sentiment_deploy")

# Configuration
REMOTE_DIR = "/root/sentiment_service"
SYSTEM_PACKAGES = [
    "python3-pip",
    "htop",
    "tmux",
    "nvtop",
    "lsof",
    "net-tools",  # for netstat
    "htop",
    "iotop",
    "ncdu",       # disk usage analyzer
    "tree",       # directory structure viewer
    "curl",
    "wget",
    "vim",
    "git"
]

@task
def check_gpu(c):
    """Check GPU availability and configuration."""
    log.info("Checking GPU availability")
    c.run("nvidia-smi", warn=True)
    gpu_info = c.run("nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv,noheader")
    log.info(f"GPU Info:\n{gpu_info.stdout}")

@task
def install_system_deps(c):
    """Install system dependencies."""
    log.info("Installing system dependencies")
    c.run("export DEBIAN_FRONTEND=noninteractive")
    c.run("apt-get update -qq")
    packages = " ".join(SYSTEM_PACKAGES)
    c.run(f"apt-get install -y -qq {packages}")

@task
def setup_python_deps(c):
    """Set up Python dependencies using Poetry and pip for PyTorch."""
    log.info("Setting up Python dependencies")
    
    # Install Poetry if not installed
    c.run('curl -sSL https://install.python-poetry.org | python3 -', warn=True)
    
    # Add Poetry to PATH
    c.run('export PATH="/root/.local/bin:$PATH"')
    c.run('source ~/.bashrc')  # Reload shell
    
    # Configure Poetry to create virtualenv in project directory
    c.run('/root/.local/bin/poetry config virtualenvs.in-project true')
    
    # Install dependencies with Poetry (except PyTorch)
    with c.cd(REMOTE_DIR):
        log.info("Installing Python packages with Poetry")
        c.run('/root/.local/bin/poetry install --only main --no-interaction --no-root')
        
        # Install PyTorch with CUDA support using pip
        log.info("Installing PyTorch with CUDA support")
        c.run('/root/.local/bin/poetry run pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
        
        # Verify PyTorch CUDA installation
        log.info("Verifying PyTorch CUDA")
        verify_script = '''
import torch
import sys
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    # Test CUDA tensor creation
    x = torch.rand(5, 3).cuda()
    print(f"Test tensor on CUDA: {x.device}")
sys.exit(0 if torch.cuda.is_available() else 1)
'''
        c.run(f'/root/.local/bin/poetry run python3 -c """{verify_script}"""')

@task
def setup_monitoring(c):
    """Set up monitoring in tmux."""
    log.info("Setting up monitoring")
    
    # Kill existing tmux session if it exists
    c.run("tmux kill-session -t sentiment_service", warn=True)
    
    # Create new tmux session
    c.run("TMUX= tmux new-session -d -s sentiment_service")
    
    # Configure tmux
    c.run("tmux set-option -t sentiment_service mouse on")
    c.run("tmux set-option -t sentiment_service history-limit 50000")
    
    # Create monitoring layout
    c.run("tmux split-window -h -t sentiment_service")
    c.run("tmux split-window -v -t sentiment_service")
    c.run("tmux select-pane -t 0")
    c.run("tmux split-window -v -t sentiment_service")
    
    # Setup monitoring tools
    c.run('tmux send-keys -t sentiment_service:0.0 "htop" C-m')
    c.run('tmux send-keys -t sentiment_service:0.1 "watch -n1 \'nvidia-smi\'" C-m')
    c.run('tmux send-keys -t sentiment_service:0.2 "nvitop" C-m')
    
    log.info("Monitoring setup complete in tmux")

@task
def start_service(c):
    """Start the sentiment analysis service."""
    log.info("Starting sentiment analysis service")
    
    with c.cd(REMOTE_DIR):
        # Start the service in the last pane using Poetry
        c.run('tmux send-keys -t sentiment_service:0.3 "cd /root/sentiment_service && /root/.local/bin/poetry run python src/analyzer.py 2>&1 | tee service.log" C-m')
        
        # Wait for service to start
        log.info("Waiting for service to start...")
        for _ in range(30):
            result = c.run("grep 'Server running at' service.log", warn=True)
            if result.ok:
                log.info("Service started successfully!")
                return
            time.sleep(1)
        
        log.error("Service failed to start within 30 seconds")
        raise Exception("Service failed to start")

@task
def deploy(c):
    """Main deployment task."""
    try:
        check_gpu(c)
        install_system_deps(c)
        setup_python_deps(c)
        setup_monitoring(c)
        start_service(c)
        
        log.info("\nDeployment successful!")
        log.info("To view the service and monitoring:")
        log.info("1. Run: tmux attach -t sentiment_service")
        log.info("2. To detach from tmux: press Ctrl+B, then D")
        log.info("3. Service is running on port 8080")
        log.info("4. View service logs: tail -f service.log")
        
    except Exception as e:
        log.error(f"Deployment failed: {str(e)}")
        raise 