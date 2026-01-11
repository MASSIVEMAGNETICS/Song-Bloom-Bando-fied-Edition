# 🚀 Windows Quick Start - Get Running in 5 Minutes!

This is the **fastest** way to get SongBloom running on Windows 10/11.

## ⚡ Super Quick Start (5 Minutes)

### Step 1: Install Prerequisites (2 minutes)

**Need these installed first:**

1. **Python 3.8-3.10** → https://www.python.org/downloads/
   - ✅ Check "Add Python to PATH" during install!
   
2. **Git** → https://git-scm.com/download/win
   - Use default options

### Step 2: Get SongBloom (1 minute)

Open **Command Prompt** and run:

```cmd
cd %USERPROFILE%\Documents
git clone https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition.git
cd Song-Bloom-Bando-fied-Edition
```

### Step 3: Run the Launcher (2 minutes)

**Just double-click:** `launch.bat`

Or from Command Prompt:
```cmd
launch.bat
```

The launcher will:
- ✅ Set up your environment automatically
- ✅ Install all dependencies
- ✅ Give you a menu to choose interface
- ✅ Launch the app!

**Choose:**
- Option 1: Conda (recommended if you have it)
- Option 2: venv (lighter, always works)
- Then select "Both" for interface installation

### Step 4: Generate Music! 🎵

1. Wait for app to open in browser
2. Click "Load Model" (first time only - downloads ~2-5GB)
3. Enter lyrics
4. Upload a style audio (optional)
5. Click "Generate Music"
6. Download your song!

---

## 📖 Need More Help?

- **Full Guide:** [WINDOWS_SETUP.md](WINDOWS_SETUP.md) - Complete installation with troubleshooting
- **Troubleshooting:** Press option 5 in the launcher menu
- **Questions?** Open an issue: https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition/issues

---

## 🛠️ Common Quick Fixes

### "Python not recognized"
```cmd
py --version
```
If that works, use `py` instead of `python`

### "Port already in use"
Close other programs or restart computer

### "Out of memory"
Lower steps to 30-40 in the interface

### "Can't download model"
Check internet connection, may take 5-15 min on first run

---

## 🎯 Tips for Best Experience

✅ **GPU Users:** Install CUDA Toolkit for 5-10x faster generation
- Download: https://developer.nvidia.com/cuda-downloads

✅ **First Run:** Be patient! Model download takes 5-15 minutes
- Only happens once, then it's cached

✅ **Quality Settings:**
- **Fast** (30 steps): 2-3 min generation, good quality
- **Balanced** (50 steps): 4-6 min, great quality
- **High** (75 steps): 6-8 min, excellent quality
- **Ultra** (100 steps): 8-10 min, best quality

✅ **Best Results:**
- Clear, descriptive lyrics work best
- 10-second style audio helps (optional)
- Try different presets to find your preference

---

## 🎵 You're Ready!

That's it! You're now ready to create amazing AI-generated music on Windows!

For advanced features like voice cloning, see the [Full Windows Setup Guide](WINDOWS_SETUP.md).

**Happy music making!** 🎶
