#!/bin/bash
# SongBloom One-Click Setup and Launch Script
# Supports both Streamlit and Gradio interfaces

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        🎵  SongBloom One-Click Setup & Launch  🎵             ║
║                                                               ║
║          Choose your interface and get started!               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check if virtual environment exists
check_venv() {
    if [ -d "venv" ]; then
        return 0
    else
        return 1
    fi
}

# Function to create conda environment
setup_conda_env() {
    echo -e "${YELLOW}Setting up Conda environment...${NC}"
    if conda env list | grep -q "^songbloom "; then
        echo -e "${GREEN}Conda environment 'songbloom' already exists.${NC}"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n songbloom -y
            conda create -n songbloom python=3.8.12 -y
        fi
    else
        conda create -n songbloom python=3.8.12 -y
    fi
    echo -e "${GREEN}✓ Conda environment ready${NC}"
}

# Function to create virtual environment
setup_venv() {
    echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
    if check_venv; then
        echo -e "${GREEN}Virtual environment already exists.${NC}"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
            python -m venv venv
        fi
    else
        python -m venv venv
    fi
    echo -e "${GREEN}✓ Virtual environment ready${NC}"
}

# Function to install dependencies
install_dependencies() {
    local interface=$1
    echo -e "${YELLOW}Installing dependencies for $interface...${NC}"
    
    if [ "$interface" == "streamlit" ]; then
        echo -e "${BLUE}Installing Streamlit requirements...${NC}"
        pip install -r requirements.txt
    else
        echo -e "${BLUE}Installing Gradio requirements...${NC}"
        cd SongBloom-master
        pip install -r requirements.txt
        cd ..
        # Also install streamlit requirements for dual support
        pip install streamlit>=1.28.0
    fi
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Function to activate environment
activate_env() {
    local env_type=$1
    
    if [ "$env_type" == "conda" ]; then
        echo -e "${YELLOW}Activating Conda environment...${NC}"
        eval "$(conda shell.bash hook)"
        conda activate songbloom
    else
        echo -e "${YELLOW}Activating virtual environment...${NC}"
        source venv/bin/activate
    fi
}

# Function to launch Streamlit
launch_streamlit() {
    echo -e "${GREEN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                 Launching Streamlit Interface                 ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo -e "${BLUE}Starting Streamlit app...${NC}"
    echo -e "${YELLOW}Access the app at: http://localhost:8501${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    streamlit run streamlit_app.py
}

# Function to launch Gradio
launch_gradio() {
    echo -e "${GREEN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                  Launching Gradio Interface                   ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo -e "${BLUE}Starting Gradio app...${NC}"
    echo -e "${YELLOW}Access the app at: http://localhost:7860${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    cd SongBloom-master
    python app.py --auto-load-model
}

# Function to launch Next-Gen X3
launch_x3() {
    echo -e "${GREEN}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║              Launching Next-Gen X3 Interface                  ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo -e "${BLUE}Starting Next-Gen X3 app...${NC}"
    echo -e "${YELLOW}Access the app at: http://localhost:7860${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    cd SongBloom-master
    python app_nextgen_x3.py --auto-load-model
}

# Main menu
show_menu() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    Choose Your Interface                      ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║                                                               ║"
    echo "║  1. 🌐 Streamlit (Modern, Cloud-Ready)                        ║"
    echo "║     - Best for: Cloud deployment, sharing                     ║"
    echo "║     - Port: 8501                                              ║"
    echo "║                                                               ║"
    echo "║  2. 🎨 Gradio (Suno-like GUI)                                 ║"
    echo "║     - Best for: Local use, familiar interface                 ║"
    echo "║     - Port: 7860                                              ║"
    echo "║                                                               ║"
    echo "║  3. 🎤 Next-Gen X3 (Voice Personas)                           ║"
    echo "║     - Best for: Voice cloning, advanced features              ║"
    echo "║     - Port: 7860                                              ║"
    echo "║                                                               ║"
    echo "║  4. ⚙️  Setup Only (Install dependencies)                     ║"
    echo "║                                                               ║"
    echo "║  5. 🚪 Exit                                                    ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Environment setup menu
show_env_menu() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                  Environment Setup Method                     ║"
    echo "╠═══════════════════════════════════════════════════════════════╣"
    echo "║                                                               ║"
    echo "║  1. 🐍 Conda (Recommended)                                    ║"
    echo "║     - Better dependency management                            ║"
    echo "║     - Isolated environment                                    ║"
    echo "║                                                               ║"
    echo "║  2. 📦 Virtual Environment (venv)                             ║"
    echo "║     - Lightweight                                             ║"
    echo "║     - Standard Python tool                                    ║"
    echo "║                                                               ║"
    echo "║  3. 🔄 Skip (Use existing environment)                        ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Main execution
main() {
    # Check if setup is needed
    SKIP_SETUP=false
    
    if [ ! -f ".songbloom_setup_complete" ]; then
        echo -e "${YELLOW}First time setup detected!${NC}"
        
        # Show environment menu
        show_env_menu
        read -p "Choose setup method (1-3): " env_choice
        
        case $env_choice in
            1)
                if ! check_conda; then
                    echo -e "${RED}Error: Conda not found. Please install Miniconda or Anaconda first.${NC}"
                    echo -e "${YELLOW}Download from: https://docs.conda.io/en/latest/miniconda.html${NC}"
                    exit 1
                fi
                setup_conda_env
                activate_env "conda"
                ;;
            2)
                setup_venv
                activate_env "venv"
                ;;
            3)
                echo -e "${YELLOW}Skipping environment setup...${NC}"
                SKIP_SETUP=true
                ;;
            *)
                echo -e "${RED}Invalid choice. Exiting.${NC}"
                exit 1
                ;;
        esac
        
        # Show interface menu for dependency installation
        if [ "$SKIP_SETUP" = false ]; then
            echo ""
            echo -e "${YELLOW}Which interface(s) do you want to use?${NC}"
            echo "1. Streamlit only"
            echo "2. Gradio only"
            echo "3. Both (recommended)"
            read -p "Choose (1-3): " dep_choice
            
            case $dep_choice in
                1)
                    install_dependencies "streamlit"
                    ;;
                2)
                    install_dependencies "gradio"
                    ;;
                3)
                    install_dependencies "gradio"  # This installs both
                    ;;
                *)
                    echo -e "${RED}Invalid choice. Installing both...${NC}"
                    install_dependencies "gradio"
                    ;;
            esac
            
            # Mark setup as complete
            touch .songbloom_setup_complete
            echo -e "${GREEN}✓ Setup complete!${NC}"
        fi
    else
        echo -e "${GREEN}✓ Environment already set up${NC}"
        
        # Try to activate existing environment
        if check_conda && conda env list | grep -q "^songbloom "; then
            activate_env "conda"
        elif check_venv; then
            activate_env "venv"
        fi
    fi
    
    # Show main menu
    while true; do
        echo ""
        show_menu
        read -p "Enter your choice (1-5): " choice
        
        case $choice in
            1)
                launch_streamlit
                ;;
            2)
                launch_gradio
                ;;
            3)
                launch_x3
                ;;
            4)
                echo -e "${YELLOW}Running setup...${NC}"
                rm -f .songbloom_setup_complete
                exec "$0"  # Restart script
                ;;
            5)
                echo -e "${GREEN}Goodbye! 👋${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please try again.${NC}"
                ;;
        esac
        
        # After app exits, return to menu
        echo ""
        echo -e "${YELLOW}App stopped. Returning to menu...${NC}"
        sleep 2
    done
}

# Run main function
main
