#!/usr/bin/env python3
"""
WiFi Vision Pro - Sample Analysis Script
Demonstrates basic usage of the WiFi signal analysis system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from wifi_signal_capture import WiFiSignalCapture, WiFiSignalPoint
    from huggingface_integration import WiFiSignalToImageAI
    import numpy as np
    import time
    
    def demo_analysis():
        print("WiFi Vision Pro - Demo Analysis")
        print("=" * 40)
        
        # Create sample signal data
        print("Creating sample signal data...")
        signals = []
        current_time = time.time()
        
        for i in range(100):
            signal = WiFiSignalPoint(
                timestamp=current_time + i * 0.1,
                frequency=2437 + (i % 3) * 25,  # Channels 6, 1, 11
                signal_strength=-50 + np.random.normal(0, 10),
                noise_level=-95,
                channel=[6, 1, 11][i % 3],
                ssid=["HomeWiFi", "OfficeNet", "GuestNetwork"][i % 3],
                bssid=f"aa:bb:cc:dd:ee:{i:02x}",
                x=np.random.uniform(0, 800),
                y=np.random.uniform(0, 600),
                phase=np.random.uniform(0, 2*np.pi),
                bandwidth=20
            )
            signals.append(signal)
        
        print(f"Generated {len(signals)} sample signals")
        
        # Analyze with AI (if available)
        try:
            print("Running AI analysis...")
            ai_converter = WiFiSignalToImageAI()
            results = ai_converter.process_wifi_signals(signals, "demo_analysis")
            
            print("AI Analysis Results:")
            for stage, result in results["processing_stages"].items():
                print(f"  {stage}: {result.get('success', 'N/A')}")
            
            print(f"Output files: {len(results.get('output_files', []))}")
            
        except Exception as e:
            print(f"AI analysis not available: {e}")
            print("This is normal if Hugging Face models are not installed")
        
        print("Demo analysis completed!")
        
    if __name__ == "__main__":
        demo_analysis()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from the demo package root directory")
