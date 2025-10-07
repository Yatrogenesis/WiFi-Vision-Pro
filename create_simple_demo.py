#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Vision Pro - Simple Demo Package Creator
Creates a working demonstration package
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def create_simple_demo():
    """Create a simple demo package"""
    print("Creating WiFi Vision Pro Demo Package")
    print("=" * 50)
    
    project_dir = Path(__file__).parent
    demo_dir = project_dir / "WiFi_Vision_Pro_Demo"
    
    # Remove existing demo directory
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    
    # Create demo structure
    demo_dir.mkdir()
    
    print("Copying source files...")
    
    # Copy main source files
    source_files = [
        "main.py",
        "advanced_gui.py", 
        "wifi_signal_capture.py",
        "huggingface_integration.py",
        "requirements.txt",
        "README.md",
        "LICENSE"
    ]
    
    for file in source_files:
        if (project_dir / file).exists():
            shutil.copy2(project_dir / file, demo_dir / file)
    
    # Create startup script
    startup_script = demo_dir / "START_WiFi_Vision_Pro.bat"
    with open(startup_script, 'w') as f:
        f.write('''@echo off
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
''')
    
    # Create simple readme
    with open(demo_dir / "QUICK_START.txt", 'w') as f:
        f.write(f'''WiFi Vision Pro v2.0.0 - Quick Start Guide
============================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

REQUIREMENTS:
- Python 3.8+
- Windows 10/11 (for this demo)
- Administrator privileges recommended

QUICK START:
1. Double-click "START_WiFi_Vision_Pro.bat"
2. Wait for dependency installation
3. Application starts automatically

FEATURES:
- Real-time WiFi signal visualization
- AI-powered signal analysis
- Cross-platform support
- Multiple visualization modes

FOR FULL INSTALLATION:
See README.md for complete setup instructions
and all available features.

TROUBLESHOOTING:
- Ensure Python is installed and in PATH
- Run as Administrator for WiFi capture
- Check Windows Defender/Firewall settings

This is a demonstration package. For commercial
licensing, contact: support@wifivisionpro.com

(c) 2025 WiFi Analysis Solutions
''')
    
    print("Creating ZIP archive...")
    
    # Create ZIP package
    zip_file = project_dir / "WiFi_Vision_Pro_v2.0.0_Demo.zip"
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(demo_dir):
            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(demo_dir.parent)
                zf.write(file_path, arc_path)
    
    # Calculate sizes
    demo_size = sum(f.stat().st_size for f in demo_dir.rglob('*') if f.is_file())
    zip_size = zip_file.stat().st_size
    
    print()
    print("DEMO PACKAGE CREATED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Demo Directory: {demo_dir}")
    print(f"Demo Size: {demo_size / (1024*1024):.1f} MB")
    print(f"ZIP Package: {zip_file}")
    print(f"ZIP Size: {zip_size / (1024*1024):.1f} MB")
    print()
    print("Package Contents:")
    print("- Complete source code")
    print("- Windows batch launcher")
    print("- Quick start guide")
    print("- All documentation")
    print()
    print("To test the demo:")
    print("1. Extract WiFi_Vision_Pro_v2.0.0_Demo.zip")
    print("2. Run START_WiFi_Vision_Pro.bat")
    print("3. Follow on-screen instructions")
    
    return True

if __name__ == "__main__":
    create_simple_demo()