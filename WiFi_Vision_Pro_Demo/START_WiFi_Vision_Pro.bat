@echo off
echo WiFi Vision Pro v2.0.0 - Advanced Signal Visualization
echo =========================================================
echo.
echo Installing dependencies...
pip install PySide6 opencv-python numpy matplotlib psutil
echo.
echo Starting WiFi Vision Pro...
python advanced_gui.py
echo.
pause
