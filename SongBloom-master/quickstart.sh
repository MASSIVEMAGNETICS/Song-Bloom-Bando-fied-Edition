#!/bin/bash
# SongBloom Next-Gen X2 Quick Start Script

set -e

echo "======================================"
echo "🎵 SongBloom Next-Gen X2 Quick Start"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_warning "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check Python version
print_status "Checking environment..."

# Create conda environment if it doesn't exist
if conda env list | grep -q "SongBloom"; then
    print_status "SongBloom environment already exists"
else
    print_status "Creating conda environment..."
    conda create -n SongBloom python=3.8.12 -y
    print_success "Environment created"
fi

# Activate environment
print_status "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SongBloom

# Install dependencies
print_status "Installing dependencies..."

# Check if PyTorch is installed
if python -c "import torch" 2>/dev/null; then
    print_status "PyTorch already installed"
else
    print_status "Installing PyTorch 2.2.0 with CUDA 11.8..."
    pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
fi

# Install main requirements
print_status "Installing SongBloom requirements..."
pip install -r requirements.txt

print_success "Installation complete!"

echo ""
echo "======================================"
echo "🚀 Choose how to run SongBloom:"
echo "======================================"
echo "1. Launch Web Interface (Suno-like GUI)"
echo "2. Start API Server"
echo "3. Run Command-Line Inference"
echo "4. Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        print_status "Launching Web Interface..."
        echo ""
        echo "The interface will open at: http://localhost:7860"
        echo "Press Ctrl+C to stop"
        echo ""
        python app.py --auto-load-model
        ;;
    2)
        print_status "Starting API Server..."
        echo ""
        echo "API documentation will be at: http://localhost:8000/docs"
        echo "Press Ctrl+C to stop"
        echo ""
        python api_server.py
        ;;
    3)
        print_status "Command-line inference requires a JSONL input file."
        echo ""
        if [ -f "example/test.jsonl" ]; then
            print_status "Found example file. Running inference..."
            python infer_optimized.py \
                --input-jsonl example/test.jsonl \
                --output-dir ./output \
                --dtype bfloat16 \
                --n-samples 1
        else
            print_warning "No example file found. Please create a JSONL input file."
            echo "Example format:"
            echo '{"idx": "song1", "lyrics": "Verse 1:\nYour lyrics here", "prompt_wav": "/path/to/prompt.wav"}'
            echo ""
            echo "Then run:"
            echo "python infer_optimized.py --input-jsonl your_file.jsonl --dtype bfloat16"
        fi
        ;;
    4)
        print_status "Exiting..."
        exit 0
        ;;
    *)
        print_warning "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
print_success "Done!"
echo ""
echo "For more options and documentation, see NEXTGEN_X2_GUIDE.md"
