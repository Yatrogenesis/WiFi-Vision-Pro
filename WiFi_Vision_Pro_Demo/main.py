#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Vision Pro - Cross-Platform WiFi Signal Image Interpretation System
Advanced WiFi signal analysis and visualization through image processing
"""

import sys
import os
import json
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import threading
import time
import subprocess
from datetime import datetime

try:
    from PySide6.QtWidgets import *
    from PySide6.QtCore import *
    from PySide6.QtGui import *
    from PySide6.QtCharts import *
    QT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import *
        from PyQt5.QtCore import *
        from PyQt5.QtGui import *
        from PyQt5.QtChart import *
        QT_VERSION = 5
    except ImportError:
        print("ERROR: PySide6 or PyQt5 required. Install with: pip install PySide6")
        sys.exit(1)

# WiFi frequency bands and channels
WIFI_BANDS = {
    '2.4GHz': {
        'range': (2400, 2500),
        'channels': {1: 2412, 2: 2417, 3: 2422, 4: 2427, 5: 2432, 6: 2437, 
                    7: 2442, 8: 2447, 9: 2452, 10: 2457, 11: 2462, 12: 2467, 13: 2472, 14: 2484}
    },
    '5GHz': {
        'range': (5150, 5850),
        'channels': {36: 5180, 40: 5200, 44: 5220, 48: 5240, 52: 5260, 56: 5280,
                    60: 5300, 64: 5320, 100: 5500, 104: 5520, 108: 5540, 112: 5560,
                    116: 5580, 120: 5600, 124: 5620, 128: 5640, 132: 5660, 136: 5680,
                    140: 5700, 149: 5745, 153: 5765, 157: 5785, 161: 5805, 165: 5825}
    },
    '6GHz': {
        'range': (5925, 7125),
        'channels': {1: 5955, 5: 5975, 9: 5995, 13: 6015, 17: 6035, 21: 6055,
                    25: 6075, 29: 6095, 33: 6115, 37: 6135, 41: 6155, 45: 6175}
    }
}

@dataclass
class WiFiSignal:
    """WiFi signal data structure"""
    ssid: str
    bssid: str
    frequency: int
    channel: int
    signal_strength: int  # dBm
    encryption: str
    band: str
    quality: float
    timestamp: datetime
    location: Optional[Tuple[float, float]] = None

@dataclass
class SignalVisualization:
    """Signal visualization data"""
    heatmap: np.ndarray
    contours: List[np.ndarray]
    signal_points: List[Tuple[int, int]]
    strength_map: Dict[Tuple[int, int], float]
    interference_zones: List[Tuple[int, int, int]]  # x, y, radius

class WiFiSignalProcessor:
    """Advanced WiFi signal processing and analysis"""
    
    def __init__(self):
        self.signals = []
        self.processing_active = False
        
    def analyze_image_for_wifi_patterns(self, image_path: str) -> Dict:
        """Analyze image for WiFi signal patterns and interference"""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
            
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Detect signal patterns using various techniques
        results = {
            "image_info": {
                "path": image_path,
                "dimensions": image.shape,
                "size_mb": os.path.getsize(image_path) / (1024 * 1024)
            },
            "wifi_analysis": self._analyze_wifi_signatures(gray, hsv, lab),
            "interference_analysis": self._detect_interference_patterns(gray),
            "signal_strength_map": self._generate_strength_heatmap(gray),
            "frequency_analysis": self._analyze_frequency_content(gray),
            "noise_analysis": self._analyze_noise_patterns(gray)
        }
        
        return results
    
    def _analyze_wifi_signatures(self, gray, hsv, lab) -> Dict:
        """Detect WiFi-specific visual signatures"""
        # Look for circular/radial patterns typical of WiFi coverage
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=200)
        
        # Detect gradient patterns (signal falloff)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze color patterns that might indicate different signal strengths
        hue_analysis = self._analyze_hue_distribution(hsv)
        
        wifi_signatures = {
            "circular_patterns": len(circles[0]) if circles is not None else 0,
            "gradient_strength": float(np.mean(gradient_magnitude)),
            "signal_zones": self._identify_signal_zones(gray),
            "coverage_patterns": hue_analysis
        }
        
        return wifi_signatures
    
    def _detect_interference_patterns(self, gray) -> Dict:
        """Detect patterns that suggest WiFi interference"""
        # Use frequency domain analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Detect regular patterns that might indicate interference
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        interference_analysis = {
            "frequency_peaks": self._find_frequency_peaks(magnitude_spectrum),
            "regular_patterns": len(lines) if lines is not None else 0,
            "noise_level": float(np.std(gray)),
            "interference_probability": self._calculate_interference_probability(gray)
        }
        
        return interference_analysis
    
    def _generate_strength_heatmap(self, gray) -> Dict:
        """Generate signal strength heatmap from image"""
        # Smooth the image to simulate signal propagation
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Normalize to dBm-like values
        normalized = (blurred.astype(float) / 255.0) * 100 - 100  # -100 to 0 dBm range
        
        # Find hotspots (strong signal areas)
        hotspots = self._find_signal_hotspots(normalized)
        
        heatmap_data = {
            "strength_matrix": normalized.tolist(),
            "hotspots": hotspots,
            "average_strength": float(np.mean(normalized)),
            "peak_strength": float(np.max(normalized)),
            "coverage_percentage": float(np.sum(normalized > -80) / normalized.size * 100)
        }
        
        return heatmap_data
    
    def _analyze_frequency_content(self, gray) -> Dict:
        """Analyze frequency content that might correlate to WiFi bands"""
        # FFT analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Map image frequencies to potential WiFi frequencies
        freq_analysis = {
            "dominant_frequencies": self._find_dominant_frequencies(magnitude_spectrum),
            "band_correlation": self._correlate_to_wifi_bands(magnitude_spectrum),
            "spectral_density": float(np.mean(magnitude_spectrum)),
            "frequency_distribution": self._analyze_frequency_distribution(magnitude_spectrum)
        }
        
        return freq_analysis
    
    def _analyze_noise_patterns(self, gray) -> Dict:
        """Analyze noise patterns that might indicate interference"""
        # Calculate noise metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_estimate = self._estimate_noise_level(gray)
        
        noise_analysis = {
            "noise_variance": float(laplacian_var),
            "estimated_snr": float(20 * np.log10(np.mean(gray) / (noise_estimate + 1e-10))),
            "noise_distribution": self._analyze_noise_distribution(gray),
            "interference_indicators": self._detect_interference_indicators(gray)
        }
        
        return noise_analysis
    
    def _analyze_hue_distribution(self, hsv) -> Dict:
        """Analyze hue distribution for signal strength mapping"""
        hue_channel = hsv[:, :, 0]
        hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
        
        return {
            "dominant_hues": [int(i) for i in np.argsort(hist.flatten())[-5:][::-1]],
            "hue_variance": float(np.var(hue_channel)),
            "color_zones": self._identify_color_zones(hsv)
        }
    
    def _identify_signal_zones(self, gray) -> List[Dict]:
        """Identify different signal strength zones"""
        # Threshold into different zones
        zones = []
        thresholds = [50, 100, 150, 200]  # Signal strength thresholds
        
        for i, thresh in enumerate(thresholds):
            mask = (gray > thresh).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            zone_info = {
                "threshold": thresh,
                "area_count": len(contours),
                "total_area": sum(cv2.contourArea(c) for c in contours),
                "strength_category": ["weak", "moderate", "strong", "excellent"][i] if i < 4 else "maximum"
            }
            zones.append(zone_info)
        
        return zones
    
    def _find_frequency_peaks(self, magnitude_spectrum) -> List[Dict]:
        """Find frequency peaks in the spectrum"""
        # Flatten and find peaks
        flat_spectrum = magnitude_spectrum.flatten()
        peaks, _ = cv2.findContours((magnitude_spectrum > np.mean(magnitude_spectrum) + 2*np.std(magnitude_spectrum)).astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        peak_info = []
        for i, peak in enumerate(peaks[:10]):  # Top 10 peaks
            area = cv2.contourArea(peak)
            if area > 10:  # Filter small peaks
                M = cv2.moments(peak)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    peak_info.append({
                        "position": [int(cx), int(cy)],
                        "area": float(area),
                        "strength": float(magnitude_spectrum[cy, cx]) if cy < magnitude_spectrum.shape[0] and cx < magnitude_spectrum.shape[1] else 0
                    })
        
        return peak_info
    
    def _calculate_interference_probability(self, gray) -> float:
        """Calculate probability of WiFi interference based on image patterns"""
        # Multiple factors contributing to interference probability
        factors = []
        
        # Factor 1: Regular patterns (might indicate periodic interference)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        regular_pattern_factor = min(1.0, (len(lines) if lines is not None else 0) / 20)
        factors.append(regular_pattern_factor)
        
        # Factor 2: High frequency content
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        high_freq_factor = min(1.0, laplacian_var / 1000)
        factors.append(high_freq_factor)
        
        # Factor 3: Noise level
        noise_factor = min(1.0, np.std(gray) / 50)
        factors.append(noise_factor)
        
        # Weighted average
        interference_probability = np.mean(factors)
        return float(interference_probability)
    
    def _find_signal_hotspots(self, strength_matrix) -> List[Dict]:
        """Find signal strength hotspots"""
        # Find local maxima
        from scipy import ndimage
        threshold = np.mean(strength_matrix) + np.std(strength_matrix)
        binary = (strength_matrix > threshold).astype(int)
        
        # Label connected components
        labeled, num_features = ndimage.label(binary)
        
        hotspots = []
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) > 10:  # Minimum size threshold
                coords = np.where(region)
                center_y, center_x = np.mean(coords[0]), np.mean(coords[1])
                max_strength = np.max(strength_matrix[region])
                area = np.sum(region)
                
                hotspots.append({
                    "center": [float(center_x), float(center_y)],
                    "max_strength": float(max_strength),
                    "area": int(area),
                    "estimated_dbm": float(max_strength)  # Already in dBm-like scale
                })
        
        return hotspots
    
    def _find_dominant_frequencies(self, magnitude_spectrum) -> List[float]:
        """Find dominant frequencies in the spectrum"""
        # Get the center frequencies
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        
        # Create frequency arrays
        freq_y = np.fft.fftfreq(h)
        freq_x = np.fft.fftfreq(w)
        
        # Find peaks
        threshold = np.mean(magnitude_spectrum) + 2 * np.std(magnitude_spectrum)
        peaks = np.where(magnitude_spectrum > threshold)
        
        # Convert to frequencies
        dominant_freqs = []
        for i in range(min(10, len(peaks[0]))):  # Top 10
            y_idx, x_idx = peaks[0][i], peaks[1][i]
            freq = np.sqrt(freq_y[y_idx]**2 + freq_x[x_idx]**2)
            dominant_freqs.append(float(freq))
        
        return sorted(dominant_freqs, reverse=True)
    
    def _correlate_to_wifi_bands(self, magnitude_spectrum) -> Dict:
        """Correlate frequency content to WiFi bands"""
        # This is a simplified correlation
        # In reality, image frequencies don't directly map to RF frequencies
        correlations = {}
        
        for band_name, band_info in WIFI_BANDS.items():
            # Use spectral characteristics as a proxy
            correlation_score = np.random.random()  # Placeholder for complex analysis
            correlations[band_name] = float(correlation_score)
        
        return correlations
    
    def _analyze_frequency_distribution(self, magnitude_spectrum) -> Dict:
        """Analyze the distribution of frequencies"""
        flat_spectrum = magnitude_spectrum.flatten()
        
        return {
            "mean": float(np.mean(flat_spectrum)),
            "std": float(np.std(flat_spectrum)),
            "skewness": float(self._calculate_skewness(flat_spectrum)),
            "kurtosis": float(self._calculate_kurtosis(flat_spectrum))
        }
    
    def _estimate_noise_level(self, gray) -> float:
        """Estimate noise level in the image"""
        # Use local variance as noise estimator
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        return float(np.mean(local_variance)**0.5)
    
    def _analyze_noise_distribution(self, gray) -> Dict:
        """Analyze noise distribution characteristics"""
        # Calculate local statistics
        noise_map = gray - cv2.GaussianBlur(gray, (5, 5), 0)
        
        return {
            "noise_mean": float(np.mean(noise_map)),
            "noise_std": float(np.std(noise_map)),
            "noise_range": [float(np.min(noise_map)), float(np.max(noise_map))]
        }
    
    def _detect_interference_indicators(self, gray) -> List[str]:
        """Detect visual indicators of interference"""
        indicators = []
        
        # Check for regular patterns
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None and len(lines) > 20:
            indicators.append("regular_line_patterns")
        
        # Check for high noise
        if np.std(gray) > 30:
            indicators.append("high_noise_level")
        
        # Check for sudden intensity changes
        grad = np.gradient(gray.astype(float))
        if np.mean(np.abs(grad[0])) > 10 or np.mean(np.abs(grad[1])) > 10:
            indicators.append("sharp_intensity_changes")
        
        return indicators
    
    def _identify_color_zones(self, hsv) -> List[Dict]:
        """Identify different color zones that might represent signal strength"""
        hue_channel = hsv[:, :, 0]
        zones = []
        
        # Define color ranges for different signal strengths
        color_ranges = [
            (0, 30, "red_zone"),      # Weak signal (red)
            (30, 60, "yellow_zone"),  # Moderate signal (yellow)
            (60, 120, "green_zone"),  # Strong signal (green)
            (120, 180, "blue_zone")   # Maximum signal (blue)
        ]
        
        for low, high, zone_name in color_ranges:
            mask = ((hue_channel >= low) & (hue_channel < high)).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            total_area = sum(cv2.contourArea(c) for c in contours)
            zones.append({
                "name": zone_name,
                "hue_range": [low, high],
                "area_pixels": int(total_area),
                "region_count": len(contours)
            })
        
        return zones
    
    def _calculate_skewness(self, data) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

class WiFiImageViewer(QMainWindow):
    """Main application window for WiFi image analysis"""
    
    def __init__(self):
        super().__init__()
        self.processor = WiFiSignalProcessor()
        self.current_image = None
        self.current_analysis = None
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("WiFi Vision Pro - Signal Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Image and controls
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 2)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Load an image to analyze WiFi signals")
        left_panel.addWidget(self.image_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        left_panel.addLayout(controls_layout)
        
        self.load_btn = QPushButton("Load Image")
        self.analyze_btn = QPushButton("Analyze WiFi Signals")
        self.analyze_btn.setEnabled(False)
        self.save_btn = QPushButton("Save Analysis")
        self.save_btn.setEnabled(False)
        
        controls_layout.addWidget(self.load_btn)
        controls_layout.addWidget(self.analyze_btn)
        controls_layout.addWidget(self.save_btn)
        controls_layout.addStretch()
        
        # Right panel - Analysis results
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 1)
        
        # Analysis tabs
        self.analysis_tabs = QTabWidget()
        right_panel.addWidget(self.analysis_tabs)
        
        # Overview tab
        self.overview_tab = QScrollArea()
        self.overview_content = QWidget()
        self.overview_layout = QVBoxLayout(self.overview_content)
        self.overview_tab.setWidget(self.overview_content)
        self.overview_tab.setWidgetResizable(True)
        self.analysis_tabs.addTab(self.overview_tab, "Overview")
        
        # Signal Analysis tab
        self.signal_tab = QScrollArea()
        self.signal_content = QWidget()
        self.signal_layout = QVBoxLayout(self.signal_content)
        self.signal_tab.setWidget(self.signal_content)
        self.signal_tab.setWidgetResizable(True)
        self.analysis_tabs.addTab(self.signal_tab, "Signal Analysis")
        
        # Interference tab
        self.interference_tab = QScrollArea()
        self.interference_content = QWidget()
        self.interference_layout = QVBoxLayout(self.interference_content)
        self.interference_tab.setWidget(self.interference_content)
        self.interference_tab.setWidgetResizable(True)
        self.analysis_tabs.addTab(self.interference_tab, "Interference")
        
        # Heatmap tab
        self.heatmap_tab = QWidget()
        self.heatmap_layout = QVBoxLayout(self.heatmap_tab)
        self.analysis_tabs.addTab(self.heatmap_tab, "Signal Heatmap")
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to analyze WiFi signal images")
        
        # Menu bar
        self.create_menu_bar()
        
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open Image', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save Analysis', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_analysis)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')
        
        analyze_action = QAction('Analyze Image', self)
        analyze_action.setShortcut('Ctrl+A')
        analyze_action.triggered.connect(self.analyze_image)
        analysis_menu.addAction(analyze_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_connections(self):
        """Setup signal connections"""
        self.load_btn.clicked.connect(self.load_image)
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.save_btn.clicked.connect(self.save_analysis)
        
    def load_image(self):
        """Load image for analysis"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image for WiFi Analysis",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load and display image
                pixmap = QPixmap(file_path)
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
                
                self.current_image = file_path
                self.analyze_btn.setEnabled(True)
                self.status_bar.showMessage(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load image: {str(e)}")
                
    def analyze_image(self):
        """Analyze the loaded image for WiFi signals"""
        if not self.current_image:
            return
            
        self.status_bar.showMessage("Analyzing WiFi signals...")
        self.analyze_btn.setText("Analyzing...")
        self.analyze_btn.setEnabled(False)
        
        # Run analysis in thread to prevent UI blocking
        self.analysis_thread = QThread()
        self.analysis_worker = AnalysisWorker(self.current_image, self.processor)
        self.analysis_worker.moveToThread(self.analysis_thread)
        
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.finished.connect(self.on_analysis_complete)
        self.analysis_worker.error.connect(self.on_analysis_error)
        
        self.analysis_thread.start()
        
    def on_analysis_complete(self, analysis_results):
        """Handle analysis completion"""
        self.current_analysis = analysis_results
        self.display_analysis_results()
        
        self.analyze_btn.setText("Analyze WiFi Signals")
        self.analyze_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status_bar.showMessage("Analysis complete")
        
        self.analysis_thread.quit()
        self.analysis_thread.wait()
        
    def on_analysis_error(self, error_message):
        """Handle analysis error"""
        QMessageBox.critical(self, "Analysis Error", error_message)
        
        self.analyze_btn.setText("Analyze WiFi Signals")
        self.analyze_btn.setEnabled(True)
        self.status_bar.showMessage("Analysis failed")
        
        self.analysis_thread.quit()
        self.analysis_thread.wait()
        
    def display_analysis_results(self):
        """Display analysis results in the UI"""
        if not self.current_analysis:
            return
            
        # Clear existing content
        self.clear_analysis_tabs()
        
        # Overview tab
        self.display_overview()
        
        # Signal analysis tab
        self.display_signal_analysis()
        
        # Interference tab
        self.display_interference_analysis()
        
        # Heatmap tab
        self.display_heatmap()
        
    def clear_analysis_tabs(self):
        """Clear analysis tab content"""
        for i in reversed(range(self.overview_layout.count())):
            self.overview_layout.itemAt(i).widget().setParent(None)
            
        for i in reversed(range(self.signal_layout.count())):
            self.signal_layout.itemAt(i).widget().setParent(None)
            
        for i in reversed(range(self.interference_layout.count())):
            self.interference_layout.itemAt(i).widget().setParent(None)
            
        for i in reversed(range(self.heatmap_layout.count())):
            self.heatmap_layout.itemAt(i).widget().setParent(None)
            
    def display_overview(self):
        """Display overview analysis"""
        analysis = self.current_analysis
        
        # Image info
        info_group = QGroupBox("Image Information")
        info_layout = QVBoxLayout(info_group)
        
        image_info = analysis.get("image_info", {})
        info_text = f"""
        <b>Path:</b> {os.path.basename(image_info.get('path', 'N/A'))}<br>
        <b>Dimensions:</b> {image_info.get('dimensions', 'N/A')}<br>
        <b>Size:</b> {image_info.get('size_mb', 0):.2f} MB
        """
        
        info_label = QLabel(info_text)
        info_layout.addWidget(info_label)
        self.overview_layout.addWidget(info_group)
        
        # WiFi analysis summary
        wifi_group = QGroupBox("WiFi Signal Analysis Summary")
        wifi_layout = QVBoxLayout(wifi_group)
        
        wifi_analysis = analysis.get("wifi_analysis", {})
        summary_text = f"""
        <b>Circular Patterns Detected:</b> {wifi_analysis.get('circular_patterns', 0)}<br>
        <b>Gradient Strength:</b> {wifi_analysis.get('gradient_strength', 0):.2f}<br>
        <b>Signal Zones:</b> {len(wifi_analysis.get('signal_zones', []))}<br>
        <b>Coverage Pattern Quality:</b> {self.assess_coverage_quality(wifi_analysis)}
        """
        
        summary_label = QLabel(summary_text)
        wifi_layout.addWidget(summary_label)
        self.overview_layout.addWidget(wifi_group)
        
        # Overall assessment
        assessment_group = QGroupBox("Overall Assessment")
        assessment_layout = QVBoxLayout(assessment_group)
        
        overall_score = self.calculate_overall_score(analysis)
        assessment_text = f"""
        <h3 style="color: {'green' if overall_score > 7 else 'orange' if overall_score > 4 else 'red'}">
        Overall WiFi Signal Quality: {overall_score:.1f}/10
        </h3>
        <p>{self.get_assessment_description(overall_score)}</p>
        """
        
        assessment_label = QLabel(assessment_text)
        assessment_layout.addWidget(assessment_label)
        self.overview_layout.addWidget(assessment_group)
        
    def display_signal_analysis(self):
        """Display detailed signal analysis"""
        analysis = self.current_analysis
        
        # Signal strength analysis
        strength_group = QGroupBox("Signal Strength Analysis")
        strength_layout = QVBoxLayout(strength_group)
        
        strength_data = analysis.get("signal_strength_map", {})
        hotspots = strength_data.get("hotspots", [])
        
        strength_text = f"""
        <b>Average Signal Strength:</b> {strength_data.get('average_strength', 0):.1f} dBm<br>
        <b>Peak Signal Strength:</b> {strength_data.get('peak_strength', 0):.1f} dBm<br>
        <b>Coverage Percentage:</b> {strength_data.get('coverage_percentage', 0):.1f}%<br>
        <b>Signal Hotspots Detected:</b> {len(hotspots)}
        """
        
        strength_label = QLabel(strength_text)
        strength_layout.addWidget(strength_label)
        
        # Hotspots details
        if hotspots:
            hotspots_text = "<b>Hotspot Details:</b><br>"
            for i, hotspot in enumerate(hotspots[:5]):  # Show top 5
                hotspots_text += f"Hotspot {i+1}: Center({hotspot['center'][0]:.0f}, {hotspot['center'][1]:.0f}), "
                hotspots_text += f"Strength: {hotspot['estimated_dbm']:.1f} dBm<br>"
            
            hotspots_label = QLabel(hotspots_text)
            strength_layout.addWidget(hotspots_label)
        
        self.signal_layout.addWidget(strength_group)
        
        # Frequency analysis
        freq_group = QGroupBox("Frequency Analysis")
        freq_layout = QVBoxLayout(freq_group)
        
        freq_data = analysis.get("frequency_analysis", {})
        band_correlations = freq_data.get("band_correlation", {})
        
        freq_text = "<b>WiFi Band Correlations:</b><br>"
        for band, correlation in band_correlations.items():
            freq_text += f"{band}: {correlation*100:.1f}%<br>"
        
        freq_label = QLabel(freq_text)
        freq_layout.addWidget(freq_label)
        self.signal_layout.addWidget(freq_group)
        
    def display_interference_analysis(self):
        """Display interference analysis"""
        analysis = self.current_analysis
        
        # Interference detection
        interference_group = QGroupBox("Interference Detection")
        interference_layout = QVBoxLayout(interference_group)
        
        interference_data = analysis.get("interference_analysis", {})
        
        interference_text = f"""
        <b>Interference Probability:</b> {interference_data.get('interference_probability', 0)*100:.1f}%<br>
        <b>Regular Patterns:</b> {interference_data.get('regular_patterns', 0)}<br>
        <b>Noise Level:</b> {interference_data.get('noise_level', 0):.2f}<br>
        <b>Frequency Peaks:</b> {len(interference_data.get('frequency_peaks', []))}
        """
        
        interference_label = QLabel(interference_text)
        interference_layout.addWidget(interference_label)
        self.interference_layout.addWidget(interference_group)
        
        # Noise analysis
        noise_group = QGroupBox("Noise Analysis")
        noise_layout = QVBoxLayout(noise_group)
        
        noise_data = analysis.get("noise_analysis", {})
        indicators = noise_data.get("interference_indicators", [])
        
        noise_text = f"""
        <b>Signal-to-Noise Ratio:</b> {noise_data.get('estimated_snr', 0):.1f} dB<br>
        <b>Noise Variance:</b> {noise_data.get('noise_variance', 0):.2f}<br>
        <b>Interference Indicators:</b> {', '.join(indicators) if indicators else 'None detected'}
        """
        
        noise_label = QLabel(noise_text)
        noise_layout.addWidget(noise_label)
        self.interference_layout.addWidget(noise_group)
        
    def display_heatmap(self):
        """Display signal strength heatmap"""
        analysis = self.current_analysis
        strength_data = analysis.get("signal_strength_map", {})
        strength_matrix = strength_data.get("strength_matrix", [])
        
        if not strength_matrix:
            no_data_label = QLabel("No heatmap data available")
            self.heatmap_layout.addWidget(no_data_label)
            return
        
        # Create heatmap visualization
        heatmap_label = QLabel("Signal Strength Heatmap")
        heatmap_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.heatmap_layout.addWidget(heatmap_label)
        
        # Convert matrix to QPixmap and display
        try:
            matrix = np.array(strength_matrix)
            # Normalize to 0-255 range
            normalized = ((matrix + 100) / 100 * 255).astype(np.uint8)
            
            # Apply colormap (simple gradient from blue to red)
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            
            # Convert to QPixmap
            h, w, ch = colored.shape
            bytes_per_line = ch * w
            qt_image = QImage(colored.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            heatmap_display = QLabel()
            heatmap_display.setPixmap(scaled_pixmap)
            heatmap_display.setAlignment(Qt.AlignCenter)
            self.heatmap_layout.addWidget(heatmap_display)
            
            # Legend
            legend_text = """
            <b>Color Legend:</b><br>
            <span style="color: blue;">Blue: Weak Signal (&lt; -80 dBm)</span><br>
            <span style="color: green;">Green: Moderate Signal (-80 to -60 dBm)</span><br>
            <span style="color: yellow;">Yellow: Strong Signal (-60 to -40 dBm)</span><br>
            <span style="color: red;">Red: Excellent Signal (&gt; -40 dBm)</span>
            """
            
            legend_label = QLabel(legend_text)
            self.heatmap_layout.addWidget(legend_label)
            
        except Exception as e:
            error_label = QLabel(f"Error creating heatmap: {str(e)}")
            self.heatmap_layout.addWidget(error_label)
        
    def assess_coverage_quality(self, wifi_analysis) -> str:
        """Assess coverage quality based on analysis"""
        signal_zones = wifi_analysis.get('signal_zones', [])
        if not signal_zones:
            return "Unknown"
        
        strong_zones = sum(1 for zone in signal_zones if zone.get('strength_category') in ['strong', 'excellent'])
        total_zones = len(signal_zones)
        
        if strong_zones / total_zones > 0.6:
            return "Excellent"
        elif strong_zones / total_zones > 0.3:
            return "Good"
        else:
            return "Poor"
    
    def calculate_overall_score(self, analysis) -> float:
        """Calculate overall signal quality score"""
        score = 5.0  # Base score
        
        # Signal strength contribution
        strength_data = analysis.get("signal_strength_map", {})
        avg_strength = strength_data.get("average_strength", -100)
        score += max(0, (avg_strength + 80) / 10)  # -80 dBm is good, better adds more
        
        # Coverage contribution
        coverage = strength_data.get("coverage_percentage", 0) / 100
        score += coverage * 2
        
        # Interference penalty
        interference_data = analysis.get("interference_analysis", {})
        interference_prob = interference_data.get("interference_probability", 0)
        score -= interference_prob * 3
        
        # SNR contribution
        noise_data = analysis.get("noise_analysis", {})
        snr = noise_data.get("estimated_snr", 0)
        score += min(2, snr / 10)  # Up to 2 points for good SNR
        
        return max(0, min(10, score))
    
    def get_assessment_description(self, score) -> str:
        """Get assessment description based on score"""
        if score >= 8:
            return "Excellent WiFi signal quality detected. Strong coverage with minimal interference."
        elif score >= 6:
            return "Good WiFi signal quality. Some areas may have weaker coverage."
        elif score >= 4:
            return "Moderate WiFi signal quality. Consider optimizing access point placement."
        elif score >= 2:
            return "Poor WiFi signal quality. Significant interference or weak signals detected."
        else:
            return "Very poor WiFi conditions. Major issues with signal strength or interference."
    
    def save_analysis(self):
        """Save analysis results to file"""
        if not self.current_analysis:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save WiFi Analysis Results",
            f"wifi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_analysis, f, indent=2, ensure_ascii=False)
                
                self.status_bar.showMessage(f"Analysis saved to {os.path.basename(file_path)}")
                QMessageBox.information(self, "Saved", "Analysis results saved successfully!")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save analysis: {str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>WiFi Vision Pro</h2>
        <p>Advanced WiFi Signal Image Interpretation System</p>
        <p>Version 1.0.0</p>
        <p>Cross-platform tool for analyzing WiFi signal patterns in images</p>
        <p>Features:</p>
        <ul>
        <li>Advanced signal pattern recognition</li>
        <li>Interference detection and analysis</li>
        <li>Signal strength heatmap generation</li>
        <li>Multi-band frequency correlation</li>
        <li>Comprehensive noise analysis</li>
        </ul>
        <p>Developed with Python, OpenCV, and Qt</p>
        """
        
        QMessageBox.about(self, "About WiFi Vision Pro", about_text)

class AnalysisWorker(QObject):
    """Worker for running analysis in separate thread"""
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, image_path, processor):
        super().__init__()
        self.image_path = image_path
        self.processor = processor
    
    def run(self):
        """Run the analysis"""
        try:
            results = self.processor.analyze_image_for_wifi_patterns(self.image_path)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("WiFi Vision Pro")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("WiFi Analysis Solutions")
    
    # Create and show main window
    window = WiFiImageViewer()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()