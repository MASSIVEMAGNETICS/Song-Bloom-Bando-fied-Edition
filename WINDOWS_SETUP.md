# 🪟 Windows 10/11 Complete Setup Guide for SongBloom

This guide provides a complete end-to-end setup and installation process for Windows 10 and Windows 11 users.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Method 1: One-Click Launcher (Recommended)](#method-1-one-click-launcher-recommended)
  - [Method 2: Manual Installation](#method-2-manual-installation)
- [GPU Setup (Optional but Recommended)](#gpu-setup-optional-but-recommended)
- [Verification](#verification)
- [First Run](#first-run)
- [Troubleshooting](#troubleshooting)
- [Common Windows Issues](#common-windows-issues)
- [Uninstallation](#uninstallation)

---

## Prerequisites

Before installing SongBloom, ensure you have the following installed on your Windows 10 or Windows 11 system:

### 1. Git for Windows

Git is required to clone the repository.

**Installation:**
1. Download Git from: https://git-scm.com/download/win
2. Run the installer (use default options)
3. Verify installation:
   ```cmd
   git --version
   ```
   Should output: `git version 2.x.x`

### 2. Python 3.8 - 3.10

Python is the runtime environment for SongBloom.

**Installation:**
1. Download Python from: https://www.python.org/downloads/
   - **Recommended:** Python 3.8.12 or Python 3.10.x
   - **Important:** During installation, check "Add Python to PATH"
2. Verify installation:
   ```cmd
   python --version
   ```
   Should output: `Python 3.8.x` or `Python 3.10.x`
   
   If the command isn't recognized, you may need to use:
   ```cmd
   py --version
   ```

### 3. Microsoft Visual C++ Build Tools (Optional but Recommended)

Some Python packages require compilation on Windows.

**Installation:**
1. Download Visual Studio Build Tools from: https://visualstudio.microsoft.com/downloads/
2. Scroll down to "All Downloads" → "Tools for Visual Studio"
3. Download "Build Tools for Visual Studio 2022"
4. Run the installer
5. Select "Desktop development with C++"
6. Click Install (requires ~6GB disk space)

**Alternative (Lighter):**
- Install only "MSVC v143 - VS 2022 C++ x64/x86 build tools"
- And "Windows 10 SDK" or "Windows 11 SDK"

### 4. Conda (Optional - Recommended for Environment Management)

Conda provides better dependency management than Python's built-in venv.

**Installation:**
1. Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html
   - Choose "Miniconda3 Windows 64-bit"
2. Run the installer
   - Check "Add Miniconda3 to my PATH environment variable" (optional but convenient)
3. Open a new Command Prompt and verify:
   ```cmd
   conda --version
   ```
   Should output: `conda 23.x.x` or similar

---

## System Requirements

### Minimum Requirements
- **OS:** Windows 10 (version 1909+) or Windows 11
- **RAM:** 8 GB
- **Storage:** 15 GB free space
- **CPU:** Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **GPU:** Optional (CPU mode supported)

### Recommended Requirements
- **OS:** Windows 10 (version 21H2+) or Windows 11
- **RAM:** 16 GB or more
- **Storage:** 30 GB free space (SSD recommended)
- **CPU:** Intel i7/AMD Ryzen 7 or better
- **GPU:** NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
  - CUDA support required for GPU acceleration

### Optimal Requirements
- **RAM:** 32 GB
- **Storage:** 50 GB free space on NVMe SSD
- **GPU:** NVIDIA RTX 4070 or better (12GB+ VRAM)
- **Network:** High-speed internet for initial model download (~2-5GB)

---

## Installation Methods

### Method 1: One-Click Launcher (Recommended)

This is the easiest method for most users. The launcher handles all setup automatically.

#### Step 1: Clone the Repository

Open Command Prompt or PowerShell and run:

```cmd
cd %USERPROFILE%\Documents
git clone https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition.git
cd Song-Bloom-Bando-fied-Edition
```

#### Step 2: Run the Launcher

Double-click `launch.bat` or run from Command Prompt:

```cmd
launch.bat
```

#### Step 3: Follow the Interactive Setup

The launcher will guide you through:

1. **Environment Setup** - Choose Conda (recommended) or venv
   - Option 1: Conda (better dependency management)
   - Option 2: venv (lightweight, uses standard Python)
   - Option 3: Skip (if you already have an environment)

2. **Dependency Installation** - Choose which interface(s) to install
   - Option 1: Streamlit only (cloud-ready interface)
   - Option 2: Gradio only (Suno-like interface)
   - Option 3: Both (recommended for flexibility)

3. **Interface Selection** - Choose how to run SongBloom
   - Option 1: Streamlit (Modern, cloud-ready)
   - Option 2: Gradio (Suno-like GUI)
   - Option 3: Next-Gen X3 (Voice personas & cloning)

The launcher will automatically:
- ✅ Create isolated environment
- ✅ Install all dependencies
- ✅ Launch the application
- ✅ Open your browser to the interface

#### What to Expect

**First Run:**
- Environment creation: 2-5 minutes
- Dependency installation: 5-15 minutes (depending on internet speed)
- Total time: ~10-20 minutes

**Subsequent Runs:**
- Launch time: 10-30 seconds
- The launcher remembers your setup!

---

### Method 2: Manual Installation

For advanced users who want more control over the installation process.

#### Step 1: Clone the Repository

```cmd
cd %USERPROFILE%\Documents
git clone https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition.git
cd Song-Bloom-Bando-fied-Edition
```

#### Step 2: Create Virtual Environment

**Option A: Using Conda (Recommended)**

```cmd
conda create -n songbloom python=3.8.12 -y
conda activate songbloom
```

**Option B: Using venv**

```cmd
python -m venv songbloom-env
songbloom-env\Scripts\activate
```

#### Step 3: Install Dependencies

**For Streamlit Interface:**

```cmd
pip install -r requirements.txt
```

**For Gradio Interface (in SongBloom-master):**

```cmd
cd SongBloom-master
pip install -r requirements.txt
cd ..
```

**For Both (Recommended):**

```cmd
pip install -r requirements.txt
cd SongBloom-master
pip install -r requirements.txt
cd ..
```

**Install PyTorch with CUDA (for GPU acceleration):**

```cmd
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

*Note: Replace `cu118` with your CUDA version (e.g., `cu121` for CUDA 12.1)*

#### Step 4: Verify Installation

```cmd
cd SongBloom-master
python test_installation.py
cd ..
```

*Note: `test_installation.py` is located in the `SongBloom-master` directory.*

---

## GPU Setup (Optional but Recommended)

GPU acceleration dramatically speeds up music generation (2-10x faster).

### NVIDIA GPU Setup

#### Step 1: Check GPU Compatibility

1. Open Command Prompt
2. Run: `wmic path win32_VideoController get name`
3. Check if you have an NVIDIA GPU (GTX 1060 or newer recommended)

#### Step 2: Install NVIDIA Drivers

1. Download latest drivers from: https://www.nvidia.com/Download/index.aspx
2. Select your GPU model
3. Download and install
4. Restart your computer

#### Step 3: Install CUDA Toolkit

1. Download CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
   - **Recommended:** CUDA 11.8 or CUDA 12.1
2. Run the installer
3. Choose "Express Installation"
4. Wait for installation (requires ~3-5GB)

#### Step 4: Verify CUDA Installation

```cmd
nvcc --version
```

Should output CUDA version information.

#### Step 5: Install PyTorch with CUDA

If you used the one-click launcher, PyTorch is already installed. To enable GPU:

```cmd
conda activate songbloom
pip uninstall torch torchvision torchaudio -y
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

*Adjust `cu118` based on your CUDA version*

#### Step 6: Verify GPU in Python

```cmd
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4090
```

### AMD GPU Setup

SongBloom primarily supports NVIDIA GPUs. AMD GPU support through ROCm is experimental and not officially supported on Windows.

**Recommendation:** Use CPU mode or consider NVIDIA GPU for best experience.

---

## Verification

After installation, verify everything is working correctly.

### Quick Verification

1. **Activate Environment:**
   ```cmd
   conda activate songbloom
   REM or
   songbloom-env\Scripts\activate
   ```

2. **Run Installation Test:**
   ```cmd
   cd SongBloom-master
   python test_installation.py
   ```

3. **Check for Errors:**
   - All core dependencies should show ✓
   - GPU should show ✓ (if CUDA is installed)
   - All project files should be found

### Expected Output

```
======================================================================
SongBloom Next-Gen X2 Installation Test
======================================================================

Core Dependencies:
----------------------------------------------------------------------
  ✓ torch                OK
  ✓ torchaudio           OK
  ✓ transformers         OK
  ✓ lightning            OK
  ✓ omegaconf            OK
  ✓ einops               OK
  ✓ huggingface_hub      OK

Next-Gen X2 Dependencies:
----------------------------------------------------------------------
  ✓ gradio               OK
  ✓ fastapi              OK
  ✓ uvicorn              OK
  ✓ accelerate           OK
  ✓ bitsandbytes         OK

Hardware:
----------------------------------------------------------------------
  ✓ CUDA                 Available (NVIDIA GeForce RTX 4090)

======================================================================
✅ Installation appears to be complete!
✅ All Next-Gen X2 features are available!
```

---

## First Run

### Using the One-Click Launcher

1. Run `launch.bat`
2. Select your preferred interface (Streamlit, Gradio, or X3)
3. Wait for the application to start
4. Your browser will open automatically
5. Follow the in-app instructions to generate music

### Manual Launch

**Streamlit Interface:**
```cmd
conda activate songbloom
streamlit run streamlit_app.py
```
Access at: http://localhost:8501

**Gradio Interface:**
```cmd
conda activate songbloom
cd SongBloom-master
python app.py --auto-load-model
```
Access at: http://localhost:7860

**Next-Gen X3 (Voice Personas):**
```cmd
conda activate songbloom
cd SongBloom-master
python app_nextgen_x3.py --auto-load-model
```
Access at: http://localhost:7860

### First-Time Model Download

**Important:** The first time you use SongBloom:
1. Click "Load Model" in the interface
2. Wait 5-15 minutes for model download (~2-5GB from HuggingFace)
3. Models are cached locally for future use
4. Subsequent runs will be instant!

### Generate Your First Song

1. **Prepare Inputs:**
   - Write lyrics (or use example lyrics)
   - Prepare a 10-second audio clip for style reference (optional)

2. **Configure Settings:**
   - Choose quality preset (Fast, Balanced, High, Ultra)
   - Adjust steps (30-100, higher = better quality but slower)
   - Set CFG coefficient (1.0-2.0, controls adherence to prompt)

3. **Generate:**
   - Click "Generate Music"
   - Wait 2-10 minutes (depending on GPU/CPU and quality settings)
   - Download and enjoy!

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Python is not recognized"

**Solution:**
1. Reinstall Python from https://www.python.org/downloads/
2. **Important:** Check "Add Python to PATH" during installation
3. Restart Command Prompt
4. Try: `py --version` instead of `python --version`

#### Issue: "Conda is not recognized"

**Solution:**
1. Close and reopen Command Prompt
2. Or manually add Conda to PATH:
   - Search "Environment Variables" in Windows
   - Edit "Path" variable
   - Add: `C:\Users\YourUsername\miniconda3\Scripts`
   - Add: `C:\Users\YourUsername\miniconda3`

#### Issue: "pip install fails with compilation errors"

**Solution:**
1. Install Visual Studio Build Tools (see Prerequisites)
2. Or try pre-compiled wheels:
   ```cmd
   pip install --upgrade pip
   pip install --only-binary :all: package-name
   ```

#### Issue: "CUDA not found" or "GPU not detected"

**Solution:**
1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Reinstall CUDA Toolkit from NVIDIA website
3. Install PyTorch with correct CUDA version:
   ```cmd
   pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
4. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

#### Issue: "Port already in use"

**Error:** `Address already in use: 8501` or `7860`

**Solution:**
1. Close other applications using the port
2. Or kill the process:
   ```cmd
   netstat -ano | findstr :8501
   taskkill /PID <PID> /F
   ```

#### Issue: "Out of memory" during generation

**Solution:**
1. Lower the number of steps (try 30 instead of 100)
2. Use INT8 quantization for lower memory usage
3. Close other applications
4. Use CPU mode if GPU has insufficient VRAM:
   ```python
   # In code, set device='cpu'
   ```

#### Issue: "Model download is very slow"

**Solution:**
1. Check internet connection
2. Use a VPN if HuggingFace is blocked
3. Manually download model:
   ```cmd
   python -c "from huggingface_hub import snapshot_download; snapshot_download('CypressYang/SongBloom')"
   ```

#### Issue: "launch.bat doesn't work"

**Solution:**
1. Right-click → "Run as Administrator"
2. Check if Python is in PATH: `python --version`
3. Try manual installation method
4. Check antivirus isn't blocking the script

#### Issue: "ImportError: DLL load failed"

**Solution:**
1. Install Visual C++ Redistributable:
   - Download from: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
   - Install both x86 and x64 versions
2. Restart computer
3. Reinstall Python packages:
   ```cmd
   pip install --force-reinstall torch torchaudio
   ```

---

## Common Windows Issues

### Antivirus/Windows Defender Interference

**Symptoms:**
- Files fail to download
- Installation hangs
- "Access denied" errors

**Solution:**
1. Add Python and repository folder to Windows Defender exclusions:
   - Settings → Update & Security → Windows Security → Virus & threat protection
   - Manage settings → Add or remove exclusions
   - Add: `C:\Users\YourUsername\Documents\Song-Bloom-Bando-fied-Edition`
   - Add: Python installation directory

### Long Path Names (260 Character Limit)

**Symptoms:**
- "Path too long" errors
- File access errors

**Solution:**
1. Enable long path support:
   - Open Registry Editor (Win + R → `regedit`)
   - Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Set `LongPathsEnabled` to `1`
   - Restart computer

2. Or clone to shorter path:
   ```cmd
   cd C:\
   git clone https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition.git songbloom
   ```

### PowerShell Execution Policy

**Symptoms:**
- Scripts don't run in PowerShell
- "Execution policy" errors

**Solution:**
1. Open PowerShell as Administrator
2. Run: `Set-ExecutionPolicy RemoteSigned`
3. Confirm with `Y`

### Network/Firewall Issues

**Symptoms:**
- Can't download models
- Connection timeouts

**Solution:**
1. Check firewall settings
2. Allow Python through firewall:
   - Settings → Update & Security → Windows Security → Firewall & network protection
   - Allow an app through firewall
   - Add Python executable
3. Try with VPN if needed

### Insufficient Disk Space

**Symptoms:**
- Installation fails partway through
- "No space left" errors

**Solution:**
1. Free up space on C: drive (need at least 15-20GB)
2. Use Disk Cleanup (Win + R → `cleanmgr`)
3. Move installation to different drive:
   ```cmd
   cd D:\
   git clone https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition.git
   ```

---

## Uninstallation

### Remove SongBloom

1. **Delete Repository:**
   ```cmd
   cd %USERPROFILE%\Documents
   rmdir /s /q Song-Bloom-Bando-fied-Edition
   ```

2. **Remove Conda Environment:**
   ```cmd
   conda env remove -n songbloom
   ```

3. **Or Remove venv:**
   ```cmd
   cd Song-Bloom-Bando-fied-Edition
   rmdir /s /q songbloom-env
   rmdir /s /q venv
   ```

4. **Clean Cache (Optional):**
   ```cmd
   rmdir /s /q %USERPROFILE%\.cache\huggingface
   ```

### Keep Models for Reinstall

Models are cached in: `%USERPROFILE%\.cache\huggingface\hub`

To keep them for faster reinstallation, don't delete this folder.

---

## Additional Resources

- **Main Documentation:** [README.md](README.md)
- **Quick Start Guide:** [QUICK_START.md](QUICK_START.md)
- **Next-Gen X2 Guide:** [SongBloom-master/NEXTGEN_X2_GUIDE.md](SongBloom-master/NEXTGEN_X2_GUIDE.md)
- **Next-Gen X3 Guide:** [SongBloom-master/NEXTGEN_X3_GUIDE.md](SongBloom-master/NEXTGEN_X3_GUIDE.md)
- **Enterprise Deployment:** [ENTERPRISE_DEPLOYMENT.md](ENTERPRISE_DEPLOYMENT.md)

## Need Help?

- **GitHub Issues:** https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition/issues
- **HuggingFace:** https://huggingface.co/CypressYang/SongBloom
- **Demo Samples:** https://cypress-yang.github.io/SongBloom_demo

---

## Success Checklist

Before generating music, ensure:

- [ ] Python 3.8-3.10 is installed and in PATH
- [ ] Git is installed
- [ ] Repository is cloned
- [ ] Virtual environment is created and activated
- [ ] All dependencies are installed (run `SongBloom-master\test_installation.py`)
- [ ] GPU is detected (if using CUDA)
- [ ] Application launches without errors
- [ ] Model is downloaded (first run only)
- [ ] Browser opens to the interface

✅ **You're ready to generate amazing music!**

---

<div align="center">

Made with ❤️ for Windows Users | SongBloom Next-Gen X3

**🎵 Happy Music Making! 🎵**

</div>
