# 🚀 One-Click Quick Start Guide

## The Easiest Way to Run SongBloom

We've created one-click launchers that handle **everything** for you:
- Environment setup (Conda or venv)
- Dependency installation
- Interface selection
- Launching the app

## Getting Started

### Linux/Mac

```bash
./launch.sh
```

### Windows

```cmd
launch.bat
```

Double-click the file or run it from the command line!

## What It Does

### First Run (Automatic Setup)

1. **Environment Setup**
   - Choose between Conda (recommended) or Python venv
   - Automatically creates isolated environment
   - Installs Python 3.8.12

2. **Dependency Installation**
   - Asks which interface(s) you want to use
   - Installs all required packages
   - Handles PyTorch, Streamlit, Gradio, and all SongBloom dependencies

3. **Ready to Launch**
   - Shows interactive menu
   - Choose your preferred interface
   - Start generating music!

### Subsequent Runs (Instant)

After the first setup, the launcher:
- ✅ Skips setup (already done!)
- ✅ Activates your environment automatically
- ✅ Shows interface selection menu
- ✅ Launches in seconds

## Interface Options

### 🌐 Streamlit (Port 8501)
**Best for:**
- Cloud deployment
- Sharing with others
- Modern, clean interface
- Easy to use

**Access at:** http://localhost:8501

### 🎨 Gradio (Port 7860)
**Best for:**
- Local use
- Familiar Suno-like interface
- Fast iteration
- Traditional UI

**Access at:** http://localhost:7860

### 🎤 Next-Gen X3 (Port 7860)
**Best for:**
- Voice cloning
- Voice personas
- Advanced features
- Professional use

**Access at:** http://localhost:7860

## Menu Features

From the launcher menu, you can:

1. **Choose Interface** - Select Streamlit, Gradio, or X3
2. **Reinstall Dependencies** - Fix broken installations
3. **Exit** - Cleanly close the launcher

After stopping an app (Ctrl+C), you'll return to the menu to try another interface!

## Troubleshooting

### "Conda not found"
**Solution:** Install Miniconda or Anaconda
- Download: https://docs.conda.io/en/latest/miniconda.html
- Or choose option 2 (venv) instead

### "Python not found"
**Solution:** Install Python 3.8 or later
- Download: https://www.python.org/downloads/
- Make sure it's in your PATH

### "pip install failed"
**Solution:** 
1. Check your internet connection
2. Try the launcher again - choose option 4 to reinstall
3. If using venv, try Conda instead

### Port already in use
**Solution:**
- Streamlit: Another app is using port 8501
- Gradio: Another app is using port 7860
- Close other apps or wait a moment

### Model download takes forever
**Note:** First run downloads ~2GB model from HuggingFace
- This is normal
- Only happens once
- Requires good internet connection

## Advanced Options

### Skip Automatic Setup

If you already have an environment:
```bash
# Activate your environment first
conda activate myenv
# or
source myenv/bin/activate

# Then run launcher
./launch.sh
# Choose option 3 to skip setup
```

### Manual Installation

If the launcher doesn't work:

```bash
# Create environment
conda create -n songbloom python=3.8.12
conda activate songbloom

# Install dependencies
pip install -r requirements.txt
cd SongBloom-master
pip install -r requirements.txt
cd ..

# Run your preferred interface
streamlit run streamlit_app.py
# or
cd SongBloom-master && python app.py --auto-load-model
```

## Switching Between Interfaces

You can easily switch:

1. Stop current app (Ctrl+C)
2. Launcher returns to menu
3. Choose different interface
4. Start immediately!

No need to close and reopen - the launcher stays running!

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended
- Python 3.8.12
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM
- 20GB disk space

### Optimal
- Python 3.8.12
- 32GB RAM
- NVIDIA GPU with 12GB+ VRAM
- SSD with 30GB+ space

## What's Installed

The launcher installs:
- **Core:** PyTorch 2.2.0, TorchAudio
- **ML Libraries:** Transformers, Lightning, Accelerate
- **Web Frameworks:** Streamlit, Gradio
- **SongBloom:** All required dependencies
- **Utils:** Audio codecs, text processing, etc.

## Support

### Getting Help
- Check error messages carefully
- Try reinstalling (option 4 in menu)
- See main README.md for detailed docs
- Open GitHub issue for bugs

### Reporting Issues
Include:
- Your OS (Windows/Linux/Mac)
- Error message (full output)
- What you were trying to do
- Which interface you chose

## Tips

💡 **First Time Users:** Choose Streamlit - it's the easiest!

💡 **Voice Cloning:** Use Next-Gen X3 interface

💡 **Quick Testing:** Use Gradio with lower steps (30-40)

💡 **Production:** Deploy Streamlit to the cloud

💡 **Local Development:** Use Gradio for fastest iteration

💡 **Sharing:** Deploy Streamlit to Streamlit Cloud

## What's Next?

After launching:

1. **Load Model** (first time only)
   - Click "Load Model" in the interface
   - Wait 2-5 minutes for download
   - Only happens once!

2. **Prepare Your Input**
   - Write lyrics
   - Find a 10-second style audio clip
   
3. **Generate**
   - Upload your style audio
   - Paste your lyrics
   - Click generate
   - Wait 2-5 minutes
   
4. **Download & Enjoy!**
   - Play in browser
   - Download FLAC file
   - Share your creation!

---

**🎵 Ready to create amazing music? Just run the launcher! 🎵**

```bash
./launch.sh    # Linux/Mac
launch.bat     # Windows
```

Happy music making! 🎶
