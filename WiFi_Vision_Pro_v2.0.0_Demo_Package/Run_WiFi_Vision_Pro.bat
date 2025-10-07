@echo off
echo WiFi Vision Pro v2.0.0 - Advanced Signal Visualization
echo =========================================================
echo.
echo Installing dependencies...
pip install -r src/requirements.txt > nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Please ensure Python and pip are installed
    pause
    exit /b 1
)

echo.
echo Starting WiFi Vision Pro...
cd src
python advanced_gui.py

echo.
echo Application closed.
pause
