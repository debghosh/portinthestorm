@echo off
REM Alphatic Portfolio Analyzer - Quick Start Script (Windows)

echo ================================================
echo Alphatic Portfolio Analyzer - Quick Start
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [*] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
    echo.
)

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo.
echo [*] Installing dependencies...
echo This may take a few minutes...
echo.
pip install -r requirements.txt --quiet

if errorlevel 1 (
    echo.
    echo [X] Error installing dependencies. Please check requirements.txt
    pause
    exit /b 1
)

echo.
echo [OK] All dependencies installed successfully!
echo.

REM Run the app
echo ================================================
echo [*] Starting Alphatic Portfolio Analyzer...
echo ================================================
echo.
echo [i] The app will open in your browser at:
echo    http://localhost:8501
echo.
echo [i] Tip: Press Ctrl+C to stop the server
echo.

streamlit run alphatic_portfolio_app.py
