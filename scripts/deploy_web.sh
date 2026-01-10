#!/bin/bash
# SongBloom Web Deployment Script
# Supports multiple hosting platforms: Streamlit Cloud, Docker, AWS, etc.

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/deployment_config.yaml"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SongBloom Web Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Parse command line arguments
PLATFORM="${1:-streamlit_cloud}"
ENV="${2:-production}"

echo -e "\n${YELLOW}Platform:${NC} $PLATFORM"
echo -e "${YELLOW}Environment:${NC} $ENV\n"

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 is not installed${NC}"
        exit 1
    fi
    
    # Check required files
    if [ ! -f "$PROJECT_ROOT/streamlit_app.py" ]; then
        echo -e "${RED}Error: streamlit_app.py not found${NC}"
        exit 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
        echo -e "${RED}Error: requirements.txt not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Prerequisites check passed${NC}"
}

# Deploy to Streamlit Cloud
deploy_streamlit_cloud() {
    echo -e "\n${YELLOW}Deploying to Streamlit Cloud...${NC}"
    
    # Ensure all required files are present
    if [ ! -f "$PROJECT_ROOT/.streamlit/config.toml" ]; then
        mkdir -p "$PROJECT_ROOT/.streamlit"
        cat > "$PROJECT_ROOT/.streamlit/config.toml" << EOF
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
EOF
        echo -e "${GREEN}✓ Created Streamlit config${NC}"
    fi
    
    # Create secrets template if not exists
    if [ ! -f "$PROJECT_ROOT/.streamlit/secrets.toml.example" ]; then
        cat > "$PROJECT_ROOT/.streamlit/secrets.toml.example" << EOF
# Example secrets file for Streamlit Cloud
# Copy this to secrets.toml and fill in your values

[huggingface]
token = "your_hf_token_here"

[model]
cache_dir = "./cache"

[security]
api_key = "your_api_key_here"
EOF
        echo -e "${GREEN}✓ Created secrets template${NC}"
    fi
    
    echo -e "\n${GREEN}Streamlit Cloud deployment files are ready!${NC}"
    echo -e "\nNext steps:"
    echo -e "1. Push your code to GitHub"
    echo -e "2. Go to https://share.streamlit.io/"
    echo -e "3. Connect your repository"
    echo -e "4. Set main file: ${YELLOW}streamlit_app.py${NC}"
    echo -e "5. Add secrets from .streamlit/secrets.toml.example"
    echo -e "6. Deploy!"
}

# Deploy using Docker
deploy_docker() {
    echo -e "\n${YELLOW}Building Docker image...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Build web application image
    docker build -t songbloom-web:latest -f- . <<EOF
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    libsndfile1 \\
    ffmpeg \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF
    
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
    
    # Run container
    echo -e "\n${YELLOW}Starting container...${NC}"
    docker run -d \\
        --name songbloom-web \\
        -p 8501:8501 \\
        -v "$PROJECT_ROOT/cache:/app/cache" \\
        -v "$PROJECT_ROOT/personas:/app/personas" \\
        --restart unless-stopped \\
        songbloom-web:latest
    
    echo -e "${GREEN}✓ Container started successfully${NC}"
    echo -e "\n${GREEN}Access the app at: http://localhost:8501${NC}"
}

# Run tests before deployment
run_tests() {
    echo -e "\n${YELLOW}Running tests...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Run Python syntax check
    python3 -m py_compile streamlit_app.py
    echo -e "${GREEN}✓ Syntax check passed${NC}"
    
    # Check imports
    python3 -c "import streamlit; import torch; import torchaudio" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Import check passed${NC}"
    else
        echo -e "${YELLOW}⚠ Some imports may not be available (this is okay for cloud deployment)${NC}"
    fi
}

# Main deployment logic
main() {
    check_prerequisites
    
    if [ "$ENV" = "production" ]; then
        run_tests
    fi
    
    case $PLATFORM in
        streamlit_cloud|streamlit)
            deploy_streamlit_cloud
            ;;
        docker)
            deploy_docker
            ;;
        *)
            echo -e "${RED}Unknown platform: $PLATFORM${NC}"
            echo -e "Supported platforms: streamlit_cloud, docker"
            exit 1
            ;;
    esac
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# Run main function
main
