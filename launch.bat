@echo off
REM SongBloom One-Click Setup and Launch Script for Windows
REM Supports both Streamlit and Gradio interfaces

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

REM Check if setup is complete
if not exist ".songbloom_setup_complete" (
    goto :setup
) else (
    goto :main_menu
)

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
    echo Error: Conda not found. Please install Miniconda or Anaconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

conda env list | find "songbloom" >nul
if not errorlevel 1 (
    echo Conda environment 'songbloom' already exists.
    set /p recreate="Do you want to recreate it? (y/N): "
    if /i "!recreate!"=="y" (
        conda env remove -n songbloom -y
        conda create -n songbloom python=3.8.12 -y
    )
) else (
    conda create -n songbloom python=3.8.12 -y
)

echo ✓ Conda environment ready
call conda activate songbloom
goto :install_deps

:setup_venv
echo Setting up Python virtual environment...
if exist "venv" (
    echo Virtual environment already exists.
    set /p recreate="Do you want to recreate it? (y/N): "
    if /i "!recreate!"=="y" (
        rmdir /s /q venv
        python -m venv venv
    )
) else (
    python -m venv venv
)

echo ✓ Virtual environment ready
call venv\Scripts\activate.bat
goto :install_deps

:install_deps
echo.
echo Which interface(s) do you want to use?
echo 1. Streamlit only
echo 2. Gradio only
echo 3. Both (recommended)
echo.
set /p dep_choice="Choose (1-3): "

if "%dep_choice%"=="1" (
    echo Installing Streamlit requirements...
    pip install -r requirements.txt
) else if "%dep_choice%"=="2" (
    echo Installing Gradio requirements...
    cd SongBloom-master
    pip install -r requirements.txt
    cd ..
    pip install streamlit>=1.28.0
) else (
    echo Installing all requirements...
    cd SongBloom-master
    pip install -r requirements.txt
    cd ..
    pip install streamlit>=1.28.0
)

echo ✓ Dependencies installed
echo. > .songbloom_setup_complete
echo ✓ Setup complete!
timeout /t 2 >nul
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
echo  5. 🚪 Exit
echo.
echo ================================================================
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto :launch_streamlit
if "%choice%"=="2" goto :launch_gradio
if "%choice%"=="3" goto :launch_x3
if "%choice%"=="4" goto :rerun_setup
if "%choice%"=="5" goto :exit
goto :invalid_choice

:launch_streamlit
cls
echo.
echo ================================================================
echo                 Launching Streamlit Interface
echo ================================================================
echo.
echo Starting Streamlit app...
echo Access the app at: http://localhost:8501
echo Press Ctrl+C to stop
echo.

REM Try to activate environment
if exist "venv" call venv\Scripts\activate.bat
conda activate songbloom 2>nul

streamlit run streamlit_app.py
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
echo Access the app at: http://localhost:7860
echo Press Ctrl+C to stop
echo.

REM Try to activate environment
if exist "venv" call venv\Scripts\activate.bat
conda activate songbloom 2>nul

cd SongBloom-master
python app.py --auto-load-model
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
echo Starting Next-Gen X3 app...
echo Access the app at: http://localhost:7860
echo Press Ctrl+C to stop
echo.

REM Try to activate environment
if exist "venv" call venv\Scripts\activate.bat
conda activate songbloom 2>nul

cd SongBloom-master
python app_nextgen_x3.py --auto-load-model
cd ..
echo.
echo App stopped. Returning to menu...
timeout /t 2 >nul
goto :main_menu

:rerun_setup
del .songbloom_setup_complete
goto :setup

:invalid_choice
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto :main_menu

:exit
echo.
echo Goodbye! 👋
timeout /t 2 >nul
exit /b 0
