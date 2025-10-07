#!/bin/bash
echo "WiFi Vision Pro v2.0.0 - Advanced Signal Visualization"
echo "========================================================="
echo

echo "Installing dependencies..."
pip3 install -r src/requirements.txt

echo
echo "Starting WiFi Vision Pro..."
cd src
python3 advanced_gui.py

echo
echo "Application closed."
read -p "Press Enter to continue..."
