@echo off
REM SongBloom One-Click Setup and Launch Script for Windows
REM Supports both Streamlit and Gradio interfaces
REM Enhanced with better error handling and prerequisites checking

setlocal enabledelayedexpansion

title SongBloom One-Click Launcher

:banner
cls
echo.
echo ================================================================
echo.
echo        🎵  SongBloom One-Click Setup ^& Launch  🎵
echo.
echo          Choose your interface and get started!
echo.
echo ================================================================
echo.

REM Check prerequisites before setup
if not exist ".songbloom_setup_complete" (
    goto :check_prerequisites
) else (
    goto :main_menu
)

:check_prerequisites
echo Checking prerequisites...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ ERROR: Python is not installed or not in PATH!
        echo.
        echo Please install Python 3.8-3.10 from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        echo.
        echo For detailed instructions, see: WINDOWS_SETUP.md
        pause
        exit /b 1
    ) else (
        echo ✓ Python found (using 'py' command)
        set PYTHON_CMD=py
    )
) else (
    echo ✓ Python found
    set PYTHON_CMD=python
)

REM Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  WARNING: Git is not installed or not in PATH!
    echo    Git is needed for updates. Download from https://git-scm.com/download/win
    echo.
) else (
    echo ✓ Git found
)

REM Check pip
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: pip is not available!
    echo Please reinstall Python with pip included.
    pause
    exit /b 1
) else (
    echo ✓ pip found
)

echo.
echo All prerequisites met! Proceeding to setup...
timeout /t 2 >nul
goto :setup

:setup
echo First time setup detected!
echo.
echo ================================================================
echo                  Environment Setup Method
echo ================================================================
echo.
echo  1. 🐍 Conda (Recommended)
echo     - Better dependency management
echo     - Isolated environment
echo.
echo  2. 📦 Virtual Environment (venv)
echo     - Lightweight
echo     - Standard Python tool
echo.
echo  3. 🔄 Skip (Use existing environment)
echo.
echo ================================================================
echo.

set /p env_choice="Choose setup method (1-3): "

if "%env_choice%"=="1" goto :setup_conda
if "%env_choice%"=="2" goto :setup_venv
if "%env_choice%"=="3" goto :install_deps
goto :invalid_choice

:setup_conda
echo Setting up Conda environment...
conda --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Conda not found!
    echo.
    echo Please install Miniconda or Anaconda first:
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    echo.
    echo Or choose option 2 (venv) instead.
    echo.
    pause
    exit /b 1
)

echo ✓ Conda detected
conda env list | find "songbloom" >nul
if not errorlevel 1 (
    echo Conda environment 'songbloom' already exists.
    set /p recreate="Do you want to recreate it? (y/N): "
    if /i "!recreate!"=="y" (
        echo Removing existing environment...
        conda env remove -n songbloom -y
        echo Creating new environment...
        conda create -n songbloom python=3.8.12 -y
    )
) else (
    echo Creating Conda environment 'songbloom' with Python 3.8.12...
    conda create -n songbloom python=3.8.12 -y
    if errorlevel 1 (
        echo ❌ ERROR: Failed to create Conda environment!
        echo Please check your Conda installation.
        pause
        exit /b 1
    )
)

echo ✓ Conda environment ready
call conda activate songbloom
if errorlevel 1 (
    echo ❌ ERROR: Failed to activate Conda environment!
    pause
    exit /b 1
)
goto :install_deps

:setup_venv
echo Setting up Python virtual environment...
if exist "venv" (
    echo Virtual environment already exists.
    set /p recreate="Do you want to recreate it? (y/N): "
    if /i "!recreate!"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q venv
        echo Creating new virtual environment...
        %PYTHON_CMD% -m venv venv
        if errorlevel 1 (
            echo ❌ ERROR: Failed to create virtual environment!
            pause
            exit /b 1
        )
    )
) else (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo ❌ ERROR: Failed to create virtual environment!
        echo Make sure Python venv module is installed.
        pause
        exit /b 1
    )
)

echo ✓ Virtual environment ready
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
goto :install_deps

:install_deps
echo.
echo ================================================================
echo                   Dependency Installation
echo ================================================================
echo.
echo Which interface(s) do you want to use?
echo.
echo 1. Streamlit only (Cloud-ready, modern interface)
echo 2. Gradio only (Suno-like GUI, familiar interface)
echo 3. Both (recommended)
echo.
set /p dep_choice="Choose (1-3): "

echo.
echo Installing dependencies... This may take 5-15 minutes.
echo Please be patient and keep your internet connection active.
echo.

if "%dep_choice%"=="1" (
    echo Installing Streamlit requirements...
    pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ ERROR: Failed to install dependencies!
        echo Check your internet connection and try again.
        pause
        exit /b 1
    )
) else if "%dep_choice%"=="2" (
    echo Installing Gradio requirements...
    pip install --upgrade pip
    cd SongBloom-master
    pip install -r requirements.txt
    if errorlevel 1 (
        cd ..
        echo ❌ ERROR: Failed to install dependencies!
        echo Check your internet connection and try again.
        pause
        exit /b 1
    )
    cd ..
    pip install streamlit>=1.28.0
) else (
    echo Installing all requirements...
    pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ ERROR: Failed to install root dependencies!
        echo Check your internet connection and try again.
        pause
        exit /b 1
    )
    cd SongBloom-master
    pip install -r requirements.txt
    if errorlevel 1 (
        cd ..
        echo ❌ ERROR: Failed to install SongBloom dependencies!
        echo Check your internet connection and try again.
        pause
        exit /b 1
    )
    cd ..
)

echo.
echo ✓ Dependencies installed successfully!
echo. > .songbloom_setup_complete
echo ✓ Setup complete!
echo.
echo TIP: On first launch, the model will be downloaded (~2-5GB).
echo      This only happens once and may take 5-15 minutes.
echo.
timeout /t 3 >nul
goto :main_menu

:main_menu
cls
echo.
echo ================================================================
echo                    Choose Your Interface
echo ================================================================
echo.
echo  1. 🌐 Streamlit (Modern, Cloud-Ready)
echo     - Best for: Cloud deployment, sharing
echo     - Port: 8501
echo.
echo  2. 🎨 Gradio (Suno-like GUI)
echo     - Best for: Local use, familiar interface
echo     - Port: 7860
echo.
echo  3. 🎤 Next-Gen X3 (Voice Personas)
echo     - Best for: Voice cloning, advanced features
echo     - Port: 7860
echo.
echo  4. ⚙️  Setup Only (Reinstall dependencies)
echo.
echo  5. ℹ️  Help ^& Troubleshooting
echo.
echo  6. 🚪 Exit
echo.
echo ================================================================
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto :launch_streamlit
if "%choice%"=="2" goto :launch_gradio
if "%choice%"=="3" goto :launch_x3
if "%choice%"=="4" goto :rerun_setup
if "%choice%"=="5" goto :show_help
if "%choice%"=="6" goto :exit
goto :invalid_choice

:launch_streamlit
cls
echo.
echo ================================================================
echo                 Launching Streamlit Interface
echo ================================================================
echo.
echo Starting Streamlit app...
echo.
echo 📍 Access the app at: http://localhost:8501
echo 📍 Your browser should open automatically
echo.
echo ⚠️  Press Ctrl+C to stop the application
echo.

REM Try to activate environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✓ Virtual environment activated
)
call conda activate songbloom 2>nul

echo Starting Streamlit...
echo.
streamlit run streamlit_app.py
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Failed to launch Streamlit!
    echo.
    echo Possible solutions:
    echo - Make sure dependencies are installed (option 4 in menu)
    echo - Check if port 8501 is already in use
    echo - See WINDOWS_SETUP.md for troubleshooting
    echo.
    pause
)
echo.
echo App stopped. Returning to menu...
timeout /t 2 >nul
goto :main_menu

:launch_gradio
cls
echo.
echo ================================================================
echo                  Launching Gradio Interface
echo ================================================================
echo.
echo Starting Gradio app...
echo.
echo 📍 Access the app at: http://localhost:7860
echo 📍 Your browser should open automatically
echo.
echo ⚠️  Press Ctrl+C to stop the application
echo ⚠️  First run will download model (~2-5GB, 5-15 minutes)
echo.

REM Try to activate environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✓ Virtual environment activated
)
call conda activate songbloom 2>nul

if not exist "SongBloom-master\app.py" (
    echo ❌ ERROR: SongBloom-master\app.py not found!
    echo Please ensure the repository is cloned correctly.
    pause
    goto :main_menu
)

cd SongBloom-master
echo Starting Gradio...
echo.
python app.py --auto-load-model
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Failed to launch Gradio!
    echo.
    echo Possible solutions:
    echo - Make sure dependencies are installed (option 4 in menu)
    echo - Check if port 7860 is already in use
    echo - See WINDOWS_SETUP.md for troubleshooting
    echo.
    pause
)
cd ..
echo.
echo App stopped. Returning to menu...
timeout /t 2 >nul
goto :main_menu

:launch_x3
cls
echo.
echo ================================================================
echo              Launching Next-Gen X3 Interface
echo ================================================================
echo.
echo Starting Next-Gen X3 app with Voice Personas...
echo.
echo 📍 Access the app at: http://localhost:7860
echo 📍 Your browser should open automatically
echo.
echo ⚠️  Press Ctrl+C to stop the application
echo ⚠️  First run will download model (~2-5GB, 5-15 minutes)
echo.

REM Try to activate environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✓ Virtual environment activated
)
call conda activate songbloom 2>nul

if not exist "SongBloom-master\app_nextgen_x3.py" (
    echo ❌ ERROR: SongBloom-master\app_nextgen_x3.py not found!
    echo Please ensure the repository is cloned correctly.
    pause
    goto :main_menu
)

cd SongBloom-master
echo Starting Next-Gen X3...
echo.
python app_nextgen_x3.py --auto-load-model
if errorlevel 1 (
    echo.
    echo ❌ ERROR: Failed to launch Next-Gen X3!
    echo.
    echo Possible solutions:
    echo - Make sure dependencies are installed (option 4 in menu)
    echo - Check if port 7860 is already in use
    echo - See WINDOWS_SETUP.md for troubleshooting
    echo.
    pause
)
cd ..
echo.
echo App stopped. Returning to menu...
timeout /t 2 >nul
goto :main_menu

:show_help
cls
echo.
echo ================================================================
echo                  Help ^& Troubleshooting
echo ================================================================
echo.
echo 📖 For detailed help, see WINDOWS_SETUP.md
echo.
echo Common Issues:
echo.
echo 1. "Python not recognized"
echo    → Install Python from https://www.python.org/downloads/
echo    → Make sure to check "Add Python to PATH" during install
echo.
echo 2. "Port already in use"
echo    → Close other applications using port 8501 or 7860
echo    → Or use: netstat -ano ^| findstr :8501
echo.
echo 3. "Out of memory"
echo    → Lower generation steps (try 30 instead of 100)
echo    → Close other applications
echo    → Consider using CPU mode
echo.
echo 4. "CUDA not found"
echo    → Install NVIDIA drivers and CUDA Toolkit
echo    → See GPU Setup section in WINDOWS_SETUP.md
echo.
echo 5. "Installation failed"
echo    → Check internet connection
echo    → Try option 4 to reinstall dependencies
echo    → See WINDOWS_SETUP.md for detailed troubleshooting
echo.
echo 📍 Full documentation: WINDOWS_SETUP.md
echo 📍 GitHub Issues: https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition/issues
echo.
echo ================================================================
pause
goto :main_menu

:rerun_setup
echo.
echo This will reinstall all dependencies.
set /p confirm="Are you sure? (y/N): "
if /i not "!confirm!"=="y" goto :main_menu
del .songbloom_setup_complete
goto :setup

:invalid_choice
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto :main_menu

:exit
cls
echo.
echo ================================================================
echo.
echo         Thank you for using SongBloom! 🎵
echo.
echo         For help and documentation:
echo         → See WINDOWS_SETUP.md
echo         → Visit https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition
echo.
echo ================================================================
echo.
timeout /t 3 >nul
exit /b 0
