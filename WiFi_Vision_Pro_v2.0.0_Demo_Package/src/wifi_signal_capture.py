#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Signal Capture and Image Generation Engine
Captures real-time WiFi signal variations and converts them to visual representations
"""

import os
import sys
import time
import json
import numpy as np
import threading
import subprocess
import platform
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import cv2
import struct

try:
    import scapy.all as scapy
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: Scapy not available. Some features will be limited.")

try:
    if platform.system() == "Windows":
        import winsound
    elif platform.system() == "Linux":
        import alsaaudio
    elif platform.system() == "Darwin":
        import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio libraries not available. Audio-based signal capture disabled.")

@dataclass
class WiFiSignalPoint:
    """Single WiFi signal measurement point"""
    timestamp: float
    frequency: float
    signal_strength: float  # dBm
    noise_level: float
    channel: int
    ssid: str
    bssid: str
    x: float  # Spatial coordinate
    y: float  # Spatial coordinate
    phase: float  # Signal phase
    bandwidth: float

@dataclass
class SignalVariation:
    """WiFi signal variation pattern"""
    frequency_range: Tuple[float, float]
    amplitude_variation: List[float]
    phase_variation: List[float]
    temporal_pattern: List[float]
    spatial_distribution: np.ndarray
    interference_markers: List[Tuple[float, float]]

class WiFiSignalCapture:
    """Real-time WiFi signal capture and analysis"""
    
    def __init__(self):
        self.capturing = False
        self.signal_data = []
        self.variation_history = []
        self.capture_thread = None
        self.image_generator = SignalToImageConverter()
        
        # Signal processing parameters
        self.sample_rate = 100  # Hz
        self.capture_duration = 60  # seconds
        self.frequency_bins = 256
        self.spatial_resolution = (800, 600)
        
        # Detection parameters
        self.rssi_threshold = -90  # dBm
        self.noise_floor = -100    # dBm
        self.channel_bandwidth = 20  # MHz
        
    def start_capture(self, interface: str = None) -> bool:
        """Start WiFi signal capture"""
        if self.capturing:
            return False
            
        self.capturing = True
        self.signal_data.clear()
        
        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(interface,),
            daemon=True
        )
        self.capture_thread.start()
        
        print(f"Started WiFi signal capture on interface: {interface or 'auto'}")
        return True
    
    def stop_capture(self):
        """Stop WiFi signal capture"""
        self.capturing = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
        
        print("WiFi signal capture stopped")
    
    def _capture_loop(self, interface: str):
        """Main capture loop"""
        capture_methods = [
            self._capture_with_scapy,
            self._capture_with_iwlist,
            self._capture_with_netsh,
            self._capture_with_iw,
            self._simulate_capture
        ]
        
        # Try different capture methods
        for method in capture_methods:
            try:
                if method(interface):
                    print(f"Using capture method: {method.__name__}")
                    return
            except Exception as e:
                print(f"Capture method {method.__name__} failed: {e}")
                continue
        
        print("All capture methods failed, using simulation")
        self._simulate_capture(interface)
    
    def _capture_with_scapy(self, interface: str) -> bool:
        """Capture using Scapy (requires root/admin)"""
        if not SCAPY_AVAILABLE:
            return False
        
        try:
            # Set interface to monitor mode (Linux/macOS)
            if platform.system() in ["Linux", "Darwin"]:
                os.system(f"sudo iwconfig {interface} mode monitor")
            
            def packet_handler(packet):
                if not self.capturing:
                    return
                    
                if packet.haslayer(scapy.Dot11):
                    self._process_dot11_packet(packet)
            
            # Start packet capture
            scapy.sniff(
                iface=interface,
                prn=packet_handler,
                stop_filter=lambda p: not self.capturing,
                timeout=1
            )
            
            return True
            
        except Exception as e:
            print(f"Scapy capture failed: {e}")
            return False
    
    def _capture_with_iwlist(self, interface: str) -> bool:
        """Capture using iwlist (Linux)"""
        if platform.system() != "Linux":
            return False
        
        try:
            while self.capturing:
                # Run iwlist scan
                result = subprocess.run(
                    ["iwlist", interface or "wlan0", "scan"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    self._parse_iwlist_output(result.stdout)
                
                time.sleep(1)  # 1 second between scans
            
            return True
            
        except Exception as e:
            print(f"iwlist capture failed: {e}")
            return False
    
    def _capture_with_netsh(self, interface: str) -> bool:
        """Capture using netsh (Windows)"""
        if platform.system() != "Windows":
            return False
        
        try:
            while self.capturing:
                # Run netsh wlan show profiles
                result = subprocess.run(
                    ["netsh", "wlan", "show", "profiles"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    self._parse_netsh_output(result.stdout)
                
                time.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"netsh capture failed: {e}")
            return False
    
    def _capture_with_iw(self, interface: str) -> bool:
        """Capture using iw (Linux)"""
        if platform.system() != "Linux":
            return False
        
        try:
            while self.capturing:
                # Run iw scan
                result = subprocess.run(
                    ["iw", "dev", interface or "wlan0", "scan"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    self._parse_iw_output(result.stdout)
                
                time.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"iw capture failed: {e}")
            return False
    
    def _simulate_capture(self, interface: str) -> bool:
        """Simulate WiFi signal capture with realistic data"""
        print("Simulating WiFi signal capture...")
        
        # Simulate different access points
        access_points = [
            {"ssid": "WiFi-2.4G", "freq": 2437, "channel": 6, "base_rssi": -45},
            {"ssid": "WiFi-5G", "freq": 5180, "channel": 36, "base_rssi": -55},
            {"ssid": "Neighbor-WiFi", "freq": 2462, "channel": 11, "base_rssi": -65},
            {"ssid": "Mobile-Hotspot", "freq": 2412, "channel": 1, "base_rssi": -75},
            {"ssid": "Enterprise-Net", "freq": 5745, "channel": 149, "base_rssi": -50},
        ]
        
        start_time = time.time()
        
        while self.capturing:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Generate realistic signal variations
            for ap in access_points:
                # Add temporal variations (interference, movement, etc.)
                time_factor = np.sin(elapsed * 0.5) * 5  # Slow variation
                noise_factor = np.random.normal(0, 3)     # Random noise
                interference = np.sin(elapsed * 2) * 8 if elapsed % 10 < 1 else 0  # Periodic interference
                
                current_rssi = ap["base_rssi"] + time_factor + noise_factor + interference
                
                # Create signal point
                signal_point = WiFiSignalPoint(
                    timestamp=current_time,
                    frequency=ap["freq"],
                    signal_strength=current_rssi,
                    noise_level=self.noise_floor + np.random.normal(0, 2),
                    channel=ap["channel"],
                    ssid=ap["ssid"],
                    bssid=self._generate_bssid(ap["ssid"]),
                    x=np.random.uniform(0, self.spatial_resolution[0]),
                    y=np.random.uniform(0, self.spatial_resolution[1]),
                    phase=np.random.uniform(0, 2*np.pi),
                    bandwidth=self.channel_bandwidth
                )
                
                self.signal_data.append(signal_point)
                
                # Generate signal variation pattern
                if len(self.signal_data) > 100:  # Need enough data for pattern
                    variation = self._extract_signal_variation(ap["ssid"])
                    if variation:
                        self.variation_history.append(variation)
            
            # Limit data size
            if len(self.signal_data) > 10000:
                self.signal_data = self.signal_data[-5000:]
            
            if len(self.variation_history) > 100:
                self.variation_history = self.variation_history[-50:]
            
            time.sleep(1.0 / self.sample_rate)  # Sample rate control
        
        return True
    
    def _process_dot11_packet(self, packet):
        """Process 802.11 packet from Scapy"""
        try:
            if packet.haslayer(scapy.Dot11Beacon) or packet.haslayer(scapy.Dot11ProbeResp):
                # Extract signal information
                rssi = packet.dBm_AntSignal if hasattr(packet, 'dBm_AntSignal') else -70
                freq = packet.Channel if hasattr(packet, 'Channel') else 2437
                
                ssid = ""
                if packet.haslayer(scapy.Dot11Elt):
                    ssid = packet[scapy.Dot11Elt].info.decode('utf-8', errors='ignore')
                
                bssid = packet.addr3 if packet.addr3 else "00:00:00:00:00:00"
                
                signal_point = WiFiSignalPoint(
                    timestamp=time.time(),
                    frequency=freq,
                    signal_strength=rssi,
                    noise_level=self.noise_floor,
                    channel=self._freq_to_channel(freq),
                    ssid=ssid,
                    bssid=bssid,
                    x=np.random.uniform(0, self.spatial_resolution[0]),
                    y=np.random.uniform(0, self.spatial_resolution[1]),
                    phase=np.random.uniform(0, 2*np.pi),
                    bandwidth=self.channel_bandwidth
                )
                
                self.signal_data.append(signal_point)
                
        except Exception as e:
            print(f"Error processing packet: {e}")
    
    def _parse_iwlist_output(self, output: str):
        """Parse iwlist scan output"""
        lines = output.split('\n')
        current_ap = {}
        
        for line in lines:
            line = line.strip()
            
            if "Address:" in line:
                if current_ap:
                    self._add_parsed_ap(current_ap)
                current_ap = {"bssid": line.split("Address: ")[1]}
            
            elif "ESSID:" in line:
                current_ap["ssid"] = line.split('ESSID:"')[1].split('"')[0]
            
            elif "Frequency:" in line:
                freq_str = line.split("Frequency:")[1].split("GHz")[0].strip()
                current_ap["frequency"] = float(freq_str) * 1000  # Convert to MHz
            
            elif "Signal level=" in line:
                rssi_str = line.split("Signal level=")[1].split("dBm")[0].strip()
                current_ap["signal_strength"] = float(rssi_str)
        
        if current_ap:
            self._add_parsed_ap(current_ap)
    
    def _parse_netsh_output(self, output: str):
        """Parse netsh wlan output"""
        # Simplified parsing - Windows netsh doesn't provide detailed signal info
        lines = output.split('\n')
        for line in lines:
            if "Profile" in line and ":" in line:
                ssid = line.split(":")[1].strip()
                # Create simulated signal data for detected profile
                signal_point = WiFiSignalPoint(
                    timestamp=time.time(),
                    frequency=2437,  # Default 2.4GHz
                    signal_strength=-60 + np.random.normal(0, 10),
                    noise_level=self.noise_floor,
                    channel=6,
                    ssid=ssid,
                    bssid=self._generate_bssid(ssid),
                    x=np.random.uniform(0, self.spatial_resolution[0]),
                    y=np.random.uniform(0, self.spatial_resolution[1]),
                    phase=np.random.uniform(0, 2*np.pi),
                    bandwidth=self.channel_bandwidth
                )
                self.signal_data.append(signal_point)
    
    def _parse_iw_output(self, output: str):
        """Parse iw scan output"""
        lines = output.split('\n')
        current_ap = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("BSS "):
                if current_ap:
                    self._add_parsed_ap(current_ap)
                current_ap = {"bssid": line.split()[1]}
            
            elif "SSID:" in line:
                current_ap["ssid"] = line.split("SSID: ")[1]
            
            elif "freq:" in line:
                current_ap["frequency"] = int(line.split("freq: ")[1])
            
            elif "signal:" in line:
                signal_str = line.split("signal: ")[1].split("dBm")[0].strip()
                current_ap["signal_strength"] = float(signal_str)
        
        if current_ap:
            self._add_parsed_ap(current_ap)
    
    def _add_parsed_ap(self, ap_data: Dict):
        """Add parsed access point data to signal data"""
        signal_point = WiFiSignalPoint(
            timestamp=time.time(),
            frequency=ap_data.get("frequency", 2437),
            signal_strength=ap_data.get("signal_strength", -70),
            noise_level=self.noise_floor,
            channel=self._freq_to_channel(ap_data.get("frequency", 2437)),
            ssid=ap_data.get("ssid", "Unknown"),
            bssid=ap_data.get("bssid", "00:00:00:00:00:00"),
            x=np.random.uniform(0, self.spatial_resolution[0]),
            y=np.random.uniform(0, self.spatial_resolution[1]),
            phase=np.random.uniform(0, 2*np.pi),
            bandwidth=self.channel_bandwidth
        )
        
        self.signal_data.append(signal_point)
    
    def _extract_signal_variation(self, ssid: str) -> Optional[SignalVariation]:
        """Extract signal variation patterns for a specific SSID"""
        # Filter data for this SSID
        ssid_data = [point for point in self.signal_data[-1000:] if point.ssid == ssid]
        
        if len(ssid_data) < 50:  # Need minimum data points
            return None
        
        # Extract variations
        timestamps = [p.timestamp for p in ssid_data]
        signal_strengths = [p.signal_strength for p in ssid_data]
        phases = [p.phase for p in ssid_data]
        frequencies = [p.frequency for p in ssid_data]
        
        # Calculate temporal patterns
        time_diffs = np.diff(timestamps)
        signal_diffs = np.diff(signal_strengths)
        temporal_pattern = signal_diffs / time_diffs  # Rate of change
        
        # Create spatial distribution
        x_coords = [p.x for p in ssid_data]
        y_coords = [p.y for p in ssid_data]
        spatial_dist = np.zeros(self.spatial_resolution)
        
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if 0 <= int(x) < self.spatial_resolution[0] and 0 <= int(y) < self.spatial_resolution[1]:
                spatial_dist[int(y), int(x)] = signal_strengths[i]
        
        # Detect interference markers (sudden signal drops)
        interference_markers = []
        for i in range(1, len(signal_strengths)):
            if signal_strengths[i] - signal_strengths[i-1] < -15:  # Sudden 15dB drop
                interference_markers.append((timestamps[i], signal_strengths[i]))
        
        return SignalVariation(
            frequency_range=(min(frequencies), max(frequencies)),
            amplitude_variation=signal_strengths,
            phase_variation=phases,
            temporal_pattern=temporal_pattern.tolist(),
            spatial_distribution=spatial_dist,
            interference_markers=interference_markers
        )
    
    def _freq_to_channel(self, freq: float) -> int:
        """Convert frequency to WiFi channel"""
        if 2412 <= freq <= 2484:  # 2.4 GHz band
            return int((freq - 2412) / 5) + 1
        elif 5170 <= freq <= 5825:  # 5 GHz band
            return int((freq - 5000) / 5)
        elif 5955 <= freq <= 7125:  # 6 GHz band
            return int((freq - 5950) / 5)
        else:
            return 0
    
    def _generate_bssid(self, ssid: str) -> str:
        """Generate realistic BSSID based on SSID"""
        import hashlib
        hash_obj = hashlib.md5(ssid.encode())
        hex_dig = hash_obj.hexdigest()[:12]
        return ':'.join(hex_dig[i:i+2] for i in range(0, 12, 2))
    
    def get_signal_variations(self) -> List[SignalVariation]:
        """Get current signal variation patterns"""
        return self.variation_history.copy()
    
    def get_current_signals(self) -> List[WiFiSignalPoint]:
        """Get current signal data"""
        return self.signal_data.copy()

class SignalToImageConverter:
    """Convert WiFi signal variations to visual images"""
    
    def __init__(self, resolution: Tuple[int, int] = (800, 600)):
        self.resolution = resolution
        self.color_maps = {
            'signal_strength': cv2.COLORMAP_JET,
            'phase': cv2.COLORMAP_HSV,
            'interference': cv2.COLORMAP_HOT,
            'temporal': cv2.COLORMAP_COOL
        }
    
    def variations_to_image(self, variations: List[SignalVariation], 
                          image_type: str = 'signal_strength') -> np.ndarray:
        """Convert signal variations to image representation"""
        
        # Create base canvas
        canvas = np.zeros(self.resolution + (3,), dtype=np.uint8)
        
        if not variations:
            return canvas
        
        if image_type == 'signal_strength':
            return self._create_strength_image(variations)
        elif image_type == 'phase':
            return self._create_phase_image(variations)
        elif image_type == 'interference':
            return self._create_interference_image(variations)
        elif image_type == 'temporal':
            return self._create_temporal_image(variations)
        elif image_type == 'composite':
            return self._create_composite_image(variations)
        else:
            return canvas
    
    def _create_strength_image(self, variations: List[SignalVariation]) -> np.ndarray:
        """Create signal strength visualization"""
        # Combine all spatial distributions
        combined_spatial = np.zeros(self.resolution)
        count_matrix = np.zeros(self.resolution)
        
        for variation in variations:
            mask = variation.spatial_distribution != 0
            combined_spatial[mask] += variation.spatial_distribution[mask]
            count_matrix[mask] += 1
        
        # Average overlapping areas
        with np.errstate(divide='ignore', invalid='ignore'):
            averaged = np.divide(combined_spatial, count_matrix, 
                               out=np.zeros_like(combined_spatial), 
                               where=count_matrix!=0)
        
        # Normalize to 0-255 range
        if np.max(averaged) > np.min(averaged):
            normalized = ((averaged - np.min(averaged)) / 
                         (np.max(averaged) - np.min(averaged)) * 255).astype(np.uint8)
        else:
            normalized = np.zeros(self.resolution, dtype=np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, self.color_maps['signal_strength'])
        
        return colored
    
    def _create_phase_image(self, variations: List[SignalVariation]) -> np.ndarray:
        """Create phase variation visualization"""
        phase_canvas = np.zeros(self.resolution, dtype=np.float32)
        
        for variation in variations:
            if not variation.phase_variation:
                continue
            
            # Map phase variations to spatial locations
            phase_array = np.array(variation.phase_variation)
            
            # Create radial pattern based on phase
            center_x, center_y = self.resolution[1] // 2, self.resolution[0] // 2
            y, x = np.ogrid[:self.resolution[0], :self.resolution[1]]
            
            # Distance from center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Phase modulation
            phase_pattern = np.mean(phase_array) + np.std(phase_array) * np.sin(distance * 0.1)
            phase_canvas += phase_pattern
        
        # Normalize to 0-255
        if np.max(phase_canvas) > np.min(phase_canvas):
            normalized = ((phase_canvas - np.min(phase_canvas)) / 
                         (np.max(phase_canvas) - np.min(phase_canvas)) * 255).astype(np.uint8)
        else:
            normalized = np.zeros(self.resolution, dtype=np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, self.color_maps['phase'])
        
        return colored
    
    def _create_interference_image(self, variations: List[SignalVariation]) -> np.ndarray:
        """Create interference pattern visualization"""
        interference_canvas = np.zeros(self.resolution, dtype=np.float32)
        
        for variation in variations:
            if not variation.interference_markers:
                continue
            
            # Create interference hotspots
            for timestamp, signal_level in variation.interference_markers:
                # Map timestamp to spatial location
                x = int((timestamp % 100) / 100 * self.resolution[1])
                y = int(abs(signal_level + 100) / 60 * self.resolution[0])  # Map dBm to y-coord
                
                # Create interference pattern around point
                if 0 <= x < self.resolution[1] and 0 <= y < self.resolution[0]:
                    # Gaussian blob for interference
                    y_grid, x_grid = np.ogrid[:self.resolution[0], :self.resolution[1]]
                    blob = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * 20**2))
                    interference_canvas += blob * abs(signal_level)
        
        # Normalize and apply colormap
        if np.max(interference_canvas) > 0:
            normalized = (interference_canvas / np.max(interference_canvas) * 255).astype(np.uint8)
        else:
            normalized = np.zeros(self.resolution, dtype=np.uint8)
        
        colored = cv2.applyColorMap(normalized, self.color_maps['interference'])
        
        return colored
    
    def _create_temporal_image(self, variations: List[SignalVariation]) -> np.ndarray:
        """Create temporal pattern visualization"""
        temporal_canvas = np.zeros(self.resolution, dtype=np.float32)
        
        for i, variation in enumerate(variations):
            if not variation.temporal_pattern:
                continue
            
            # Map temporal patterns to image
            pattern = np.array(variation.temporal_pattern)
            
            # Create horizontal bands for each variation
            band_height = self.resolution[0] // max(len(variations), 1)
            y_start = i * band_height
            y_end = min(y_start + band_height, self.resolution[0])
            
            # Interpolate pattern to image width
            if len(pattern) > 1:
                interpolated = np.interp(
                    np.linspace(0, len(pattern)-1, self.resolution[1]),
                    range(len(pattern)),
                    pattern
                )
                
                # Fill band with pattern
                for y in range(y_start, y_end):
                    temporal_canvas[y, :] = interpolated
        
        # Normalize and apply colormap
        if np.max(temporal_canvas) > np.min(temporal_canvas):
            normalized = ((temporal_canvas - np.min(temporal_canvas)) / 
                         (np.max(temporal_canvas) - np.min(temporal_canvas)) * 255).astype(np.uint8)
        else:
            normalized = np.zeros(self.resolution, dtype=np.uint8)
        
        colored = cv2.applyColorMap(normalized, self.color_maps['temporal'])
        
        return colored
    
    def _create_composite_image(self, variations: List[SignalVariation]) -> np.ndarray:
        """Create composite visualization combining multiple aspects"""
        # Create individual components
        strength_img = self._create_strength_image(variations)
        phase_img = self._create_phase_image(variations)
        interference_img = self._create_interference_image(variations)
        
        # Combine with different weights
        composite = (
            strength_img.astype(np.float32) * 0.5 +
            phase_img.astype(np.float32) * 0.3 +
            interference_img.astype(np.float32) * 0.2
        )
        
        # Ensure valid range
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        
        return composite
    
    def create_real_time_image(self, signal_points: List[WiFiSignalPoint]) -> np.ndarray:
        """Create real-time image from current signal points"""
        canvas = np.zeros(self.resolution, dtype=np.float32)
        
        # Plot signal strength at each point location
        for point in signal_points[-100:]:  # Use last 100 points
            x, y = int(point.x), int(point.y)
            if 0 <= x < self.resolution[1] and 0 <= y < self.resolution[0]:
                # Map signal strength to intensity (dBm to 0-1 range)
                intensity = max(0, (point.signal_strength + 100) / 50)  # -100 to -50 dBm range
                canvas[y, x] = max(canvas[y, x], intensity)
        
        # Apply Gaussian blur for signal propagation effect
        blurred = cv2.GaussianBlur(canvas, (21, 21), 0)
        
        # Normalize and apply colormap
        normalized = (blurred * 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return colored

def main():
    """Test the WiFi signal capture system"""
    print("WiFi Signal Capture and Image Generation Test")
    print("=" * 50)
    
    # Create capture system
    capture_system = WiFiSignalCapture()
    
    # Start capture
    print("Starting WiFi signal capture...")
    capture_system.start_capture()
    
    try:
        # Let it capture for a few seconds
        time.sleep(10)
        
        # Get variations and create images
        variations = capture_system.get_signal_variations()
        signals = capture_system.get_current_signals()
        
        print(f"Captured {len(signals)} signal points")
        print(f"Extracted {len(variations)} variation patterns")
        
        # Create image converter
        converter = SignalToImageConverter()
        
        # Generate different types of images
        if variations:
            strength_img = converter.variations_to_image(variations, 'signal_strength')
            phase_img = converter.variations_to_image(variations, 'phase')
            interference_img = converter.variations_to_image(variations, 'interference')
            temporal_img = converter.variations_to_image(variations, 'temporal')
            composite_img = converter.variations_to_image(variations, 'composite')
            
            # Save images
            cv2.imwrite("wifi_strength.png", strength_img)
            cv2.imwrite("wifi_phase.png", phase_img)
            cv2.imwrite("wifi_interference.png", interference_img)
            cv2.imwrite("wifi_temporal.png", temporal_img)
            cv2.imwrite("wifi_composite.png", composite_img)
            
            print("Images saved:")
            print("- wifi_strength.png (Signal strength)")
            print("- wifi_phase.png (Phase variations)")
            print("- wifi_interference.png (Interference patterns)")
            print("- wifi_temporal.png (Temporal patterns)")
            print("- wifi_composite.png (Composite view)")
        
        # Create real-time image
        if signals:
            realtime_img = converter.create_real_time_image(signals)
            cv2.imwrite("wifi_realtime.png", realtime_img)
            print("- wifi_realtime.png (Real-time signals)")
        
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    finally:
        # Stop capture
        capture_system.stop_capture()
        print("WiFi signal capture stopped")

if __name__ == "__main__":
    main()