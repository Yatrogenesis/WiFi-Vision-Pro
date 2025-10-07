#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Vision Pro - Advanced Multi-Platform GUI with AI Integration
Real-time WiFi signal visualization and interpretation using AI models
"""

import sys
import os
import json
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

try:
    from PySide6.QtWidgets import *
    from PySide6.QtCore import *
    from PySide6.QtGui import *
    from PySide6.QtMultimedia import *
    from PySide6.QtCharts import *
    QT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import *
        from PyQt5.QtCore import *
        from PyQt5.QtGui import *
        from PyQt5.QtMultimedia import *
        from PyQt5.QtChart import *
        QT_VERSION = 5
    except ImportError:
        print("ERROR: PySide6 or PyQt5 required. Install with: pip install PySide6")
        sys.exit(1)

import cv2
from wifi_signal_capture import WiFiSignalCapture, SignalToImageConverter
from huggingface_integration import WiFiSignalToImageAI, HuggingFaceSignalProcessor

class RealTimeSignalThread(QThread):
    """Thread for real-time signal capture and processing"""
    signal_updated = Signal(list)
    image_generated = Signal(np.ndarray, str)
    ai_processing_complete = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.capture_system = WiFiSignalCapture()
        self.ai_converter = WiFiSignalToImageAI()
        self.running = False
        self.interface = None
        self.update_interval = 2.0  # seconds
        
    def start_capture(self, interface: str = None):
        """Start signal capture"""
        self.interface = interface
        self.running = True
        if not self.isRunning():
            self.start()
    
    def stop_capture(self):
        """Stop signal capture"""
        self.running = False
        self.capture_system.stop_capture()
        if self.isRunning():
            self.wait(5000)
    
    def run(self):
        """Main thread execution"""
        try:
            # Start WiFi capture
            if not self.capture_system.start_capture(self.interface):
                self.error_occurred.emit("Failed to start WiFi signal capture")
                return
            
            last_ai_processing = 0
            ai_processing_interval = 10.0  # Process with AI every 10 seconds
            
            while self.running:
                # Get current signals
                current_signals = self.capture_system.get_current_signals()
                
                if current_signals:
                    # Emit signal data for real-time updates
                    self.signal_updated.emit(current_signals.copy())
                    
                    # Generate real-time visualization
                    converter = SignalToImageConverter()
                    realtime_image = converter.create_real_time_image(current_signals)
                    self.image_generated.emit(realtime_image, "realtime")
                    
                    # Periodic AI processing
                    current_time = time.time()
                    if current_time - last_ai_processing > ai_processing_interval:
                        try:
                            # Process with AI models
                            ai_results = self.ai_converter.process_wifi_signals(
                                current_signals[-100:],  # Last 100 points
                                f"realtime_{int(current_time)}"
                            )
                            self.ai_processing_complete.emit(ai_results)
                            last_ai_processing = current_time
                            
                        except Exception as e:
                            print(f"AI processing error: {e}")
                
                # Wait for next update
                self.msleep(int(self.update_interval * 1000))
                
        except Exception as e:
            self.error_occurred.emit(f"Capture thread error: {str(e)}")
        finally:
            self.capture_system.stop_capture()

class SignalVisualizationWidget(QWidget):
    """Advanced signal visualization widget"""
    
    def __init__(self):
        super().__init__()
        self.current_signals = []
        self.current_image = None
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the visualization UI"""
        layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.reset_view_btn = QPushButton("Reset View")
        self.save_image_btn = QPushButton("Save Image")
        
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.save_image_btn.clicked.connect(self.save_current_image)
        
        toolbar.addWidget(self.zoom_in_btn)
        toolbar.addWidget(self.zoom_out_btn)
        toolbar.addWidget(self.reset_view_btn)
        toolbar.addWidget(self.save_image_btn)
        toolbar.addStretch()
        
        layout.addLayout(toolbar)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("WiFi signal visualization will appear here")
        
        # Enable mouse events
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.wheelEvent = self.wheel_event
        
        # Scroll area for pan/zoom
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
    def update_visualization(self, image: np.ndarray, image_type: str = ""):
        """Update the visualization with new image"""
        self.current_image = image.copy()
        
        # Convert numpy array to QPixmap
        if len(image.shape) == 3:
            # Color image
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            # Convert BGR to RGB
            q_image = q_image.rgbSwapped()
        else:
            # Grayscale image
            height, width = image.shape
            q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        
        # Apply zoom
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            pixmap.size() * self.zoom_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
        
        # Add image type indicator
        if image_type:
            self.setToolTip(f"Visualization Type: {image_type}")
    
    def zoom_in(self):
        """Zoom in on the image"""
        self.zoom_factor = min(self.zoom_factor * 1.5, 5.0)
        if self.current_image is not None:
            self.update_visualization(self.current_image)
    
    def zoom_out(self):
        """Zoom out of the image"""
        self.zoom_factor = max(self.zoom_factor / 1.5, 0.2)
        if self.current_image is not None:
            self.update_visualization(self.current_image)
    
    def reset_view(self):
        """Reset zoom and pan"""
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        if self.current_image is not None:
            self.update_visualization(self.current_image)
    
    def save_current_image(self):
        """Save current visualization"""
        if self.current_image is None:
            QMessageBox.information(self, "No Image", "No visualization to save")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save WiFi Visualization",
            f"wifi_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if filename:
            cv2.imwrite(filename, self.current_image)
            QMessageBox.information(self, "Saved", f"Image saved to {filename}")
    
    def mouse_press_event(self, event):
        """Handle mouse press for panning"""
        if event.button() == Qt.LeftButton:
            self.last_pan_point = event.pos()
    
    def mouse_move_event(self, event):
        """Handle mouse move for panning"""
        if event.buttons() == Qt.LeftButton and hasattr(self, 'last_pan_point'):
            delta = event.pos() - self.last_pan_point
            self.pan_offset += delta
            self.last_pan_point = event.pos()
    
    def wheel_event(self, event):
        """Handle mouse wheel for zooming"""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

class SignalAnalysisPanel(QWidget):
    """Panel for detailed signal analysis"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup analysis panel UI"""
        layout = QVBoxLayout(self)
        
        # Analysis tabs
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Real-time stats tab
        self.stats_tab = self.create_stats_tab()
        self.tab_widget.addTab(self.stats_tab, "Real-time Stats")
        
        # AI Analysis tab
        self.ai_tab = self.create_ai_tab()
        self.tab_widget.addTab(self.ai_tab, "AI Analysis")
        
        # Network detection tab
        self.network_tab = self.create_network_tab()
        self.tab_widget.addTab(self.network_tab, "Networks")
        
        # Charts tab
        self.charts_tab = self.create_charts_tab()
        self.tab_widget.addTab(self.charts_tab, "Charts")
    
    def create_stats_tab(self) -> QWidget:
        """Create real-time statistics tab"""
        tab = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Signal count
        self.signal_count_label = QLabel("Signals Detected: 0")
        self.signal_count_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.signal_count_label)
        
        # Average signal strength
        self.avg_strength_label = QLabel("Average Signal Strength: N/A")
        layout.addWidget(self.avg_strength_label)
        
        # Frequency distribution
        self.freq_dist_label = QLabel("Frequency Distribution:")
        self.freq_dist_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.freq_dist_label)
        
        self.band_2_4_label = QLabel("2.4 GHz Band: 0 signals")
        self.band_5_0_label = QLabel("5.0 GHz Band: 0 signals")
        self.band_6_0_label = QLabel("6.0 GHz Band: 0 signals")
        
        layout.addWidget(self.band_2_4_label)
        layout.addWidget(self.band_5_0_label)
        layout.addWidget(self.band_6_0_label)
        
        # Interference detection
        self.interference_label = QLabel("Interference Level: Low")
        self.interference_label.setStyleSheet("color: green;")
        layout.addWidget(self.interference_label)
        
        layout.addStretch()
        
        tab.setWidget(content)
        tab.setWidgetResizable(True)
        return tab
    
    def create_ai_tab(self) -> QWidget:
        """Create AI analysis tab"""
        tab = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        self.ai_status_label = QLabel("AI Analysis: Ready")
        self.ai_status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.ai_status_label)
        
        self.ai_results_text = QTextEdit()
        self.ai_results_text.setReadOnly(True)
        self.ai_results_text.setMinimumHeight(200)
        self.ai_results_text.setPlainText("AI analysis results will appear here...")
        layout.addWidget(self.ai_results_text)
        
        # Processing stages
        self.processing_stages = QTreeWidget()
        self.processing_stages.setHeaderLabels(["Stage", "Status", "Details"])
        layout.addWidget(self.processing_stages)
        
        layout.addStretch()
        
        tab.setWidget(content)
        tab.setWidgetResizable(True)
        return tab
    
    def create_network_tab(self) -> QWidget:
        """Create network detection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Network list
        self.network_table = QTableWidget()
        self.network_table.setColumnCount(6)
        self.network_table.setHorizontalHeaderLabels([
            "SSID", "BSSID", "Channel", "Frequency", "Signal Strength", "Last Seen"
        ])
        self.network_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.network_table)
        
        return tab
    
    def create_charts_tab(self) -> QWidget:
        """Create charts tab with signal visualizations"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        try:
            # Create chart view
            chart_view = QChartView()
            
            # Signal strength over time chart
            self.signal_chart = QChart()
            self.signal_chart.setTitle("Signal Strength Over Time")
            
            self.signal_series = QLineSeries()
            self.signal_series.setName("Signal Strength (dBm)")
            
            self.signal_chart.addSeries(self.signal_series)
            self.signal_chart.createDefaultAxes()
            
            chart_view.setChart(self.signal_chart)
            chart_view.setRenderHint(QPainter.Antialiasing)
            
            layout.addWidget(chart_view)
            
        except Exception as e:
            print(f"Charts not available: {e}")
            layout.addWidget(QLabel("Charts not available in this Qt version"))
        
        return tab
    
    def update_stats(self, signals: List):
        """Update real-time statistics"""
        if not signals:
            return
        
        # Update signal count
        self.signal_count_label.setText(f"Signals Detected: {len(signals)}")
        
        # Calculate average signal strength
        strengths = [s.signal_strength for s in signals]
        avg_strength = sum(strengths) / len(strengths)
        self.avg_strength_label.setText(f"Average Signal Strength: {avg_strength:.1f} dBm")
        
        # Frequency distribution
        band_2_4 = len([s for s in signals if 2400 <= s.frequency <= 2500])
        band_5_0 = len([s for s in signals if 5000 <= s.frequency <= 6000])
        band_6_0 = len([s for s in signals if 5925 <= s.frequency <= 7125])
        
        self.band_2_4_label.setText(f"2.4 GHz Band: {band_2_4} signals")
        self.band_5_0_label.setText(f"5.0 GHz Band: {band_5_0} signals")
        self.band_6_0_label.setText(f"6.0 GHz Band: {band_6_0} signals")
        
        # Interference detection (simplified)
        signal_variance = np.var(strengths) if len(strengths) > 1 else 0
        if signal_variance > 100:
            self.interference_label.setText("Interference Level: High")
            self.interference_label.setStyleSheet("color: red;")
        elif signal_variance > 50:
            self.interference_label.setText("Interference Level: Medium")
            self.interference_label.setStyleSheet("color: orange;")
        else:
            self.interference_label.setText("Interference Level: Low")
            self.interference_label.setStyleSheet("color: green;")
        
        # Update network table
        self.update_network_table(signals)
        
        # Update charts
        self.update_charts(signals)
    
    def update_network_table(self, signals: List):
        """Update network detection table"""
        # Group signals by SSID
        networks = {}
        for signal in signals:
            if signal.ssid not in networks:
                networks[signal.ssid] = signal
            else:
                # Keep the one with strongest signal
                if signal.signal_strength > networks[signal.ssid].signal_strength:
                    networks[signal.ssid] = signal
        
        # Update table
        self.network_table.setRowCount(len(networks))
        for row, (ssid, signal) in enumerate(networks.items()):
            self.network_table.setItem(row, 0, QTableWidgetItem(signal.ssid))
            self.network_table.setItem(row, 1, QTableWidgetItem(signal.bssid))
            self.network_table.setItem(row, 2, QTableWidgetItem(str(signal.channel)))
            self.network_table.setItem(row, 3, QTableWidgetItem(f"{signal.frequency} MHz"))
            self.network_table.setItem(row, 4, QTableWidgetItem(f"{signal.signal_strength:.1f} dBm"))
            self.network_table.setItem(row, 5, QTableWidgetItem(
                datetime.fromtimestamp(signal.timestamp).strftime("%H:%M:%S")
            ))
    
    def update_charts(self, signals: List):
        """Update signal strength charts"""
        try:
            # Clear existing data
            self.signal_series.clear()
            
            # Add recent signal data (last 100 points)
            recent_signals = signals[-100:] if len(signals) > 100 else signals
            
            for i, signal in enumerate(recent_signals):
                self.signal_series.append(i, signal.signal_strength)
            
            # Update chart axes
            if hasattr(self.signal_chart, 'axes'):
                axes = self.signal_chart.axes()
                if len(axes) >= 2:
                    axes[1].setRange(min(-100, min(s.signal_strength for s in recent_signals)), 
                                   max(-30, max(s.signal_strength for s in recent_signals)))
                
        except Exception as e:
            print(f"Chart update error: {e}")
    
    def update_ai_analysis(self, ai_results: Dict):
        """Update AI analysis results"""
        self.ai_status_label.setText("AI Analysis: Complete")
        
        # Format results for display
        results_text = f"Analysis completed at: {datetime.now().strftime('%H:%M:%S')}\n\n"
        
        for stage, result in ai_results.get("processing_stages", {}).items():
            results_text += f"{stage.upper()}:\n"
            if isinstance(result, dict):
                for key, value in result.items():
                    results_text += f"  {key}: {value}\n"
            else:
                results_text += f"  {result}\n"
            results_text += "\n"
        
        self.ai_results_text.setPlainText(results_text)
        
        # Update processing stages tree
        self.processing_stages.clear()
        for stage, result in ai_results.get("processing_stages", {}).items():
            stage_item = QTreeWidgetItem([stage, "Complete", ""])
            if isinstance(result, dict):
                for key, value in result.items():
                    detail_item = QTreeWidgetItem([str(key), "", str(value)])
                    stage_item.addChild(detail_item)
            self.processing_stages.addTopLevelItem(stage_item)
        
        self.processing_stages.expandAll()

class WiFiVisionProMainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.signal_thread = None
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup main window UI"""
        self.setWindowTitle("WiFi Vision Pro - Advanced Signal Visualization with AI")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Visualization
        self.visualization_widget = SignalVisualizationWidget()
        main_layout.addWidget(self.visualization_widget, 2)
        
        # Right panel - Analysis
        self.analysis_panel = SignalAnalysisPanel()
        main_layout.addWidget(self.analysis_panel, 1)
        
        # Setup menu and toolbar
        self.create_menu_bar()
        self.create_toolbar()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - WiFi Vision Pro with AI Integration")
        
        # Apply dark theme
        self.apply_dark_theme()
        
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_action = QAction('Load Signal Data', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_signal_data)
        file_menu.addAction(load_action)
        
        save_session_action = QAction('Save Session', self)
        save_session_action.setShortcut('Ctrl+S')
        save_session_action.triggered.connect(self.save_session)
        file_menu.addAction(save_session_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Capture menu
        capture_menu = menubar.addMenu('Capture')
        
        start_action = QAction('Start Real-time Capture', self)
        start_action.setShortcut('Ctrl+R')
        start_action.triggered.connect(self.start_realtime_capture)
        capture_menu.addAction(start_action)
        
        stop_action = QAction('Stop Capture', self)
        stop_action.setShortcut('Ctrl+T')
        stop_action.triggered.connect(self.stop_realtime_capture)
        capture_menu.addAction(stop_action)
        
        # AI menu
        ai_menu = menubar.addMenu('AI Analysis')
        
        process_action = QAction('Process with AI', self)
        process_action.setShortcut('Ctrl+A')
        process_action.triggered.connect(self.process_with_ai)
        ai_menu.addAction(process_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create application toolbar"""
        toolbar = self.addToolBar('Main')
        
        # Start/Stop buttons
        self.start_btn = QAction('▶ Start Capture', self)
        self.start_btn.triggered.connect(self.start_realtime_capture)
        toolbar.addAction(self.start_btn)
        
        self.stop_btn = QAction('⏹ Stop Capture', self)
        self.stop_btn.triggered.connect(self.stop_realtime_capture)
        self.stop_btn.setEnabled(False)
        toolbar.addAction(self.stop_btn)
        
        toolbar.addSeparator()
        
        # AI processing button
        self.ai_btn = QAction('🤖 AI Analysis', self)
        self.ai_btn.triggered.connect(self.process_with_ai)
        toolbar.addAction(self.ai_btn)
        
        toolbar.addSeparator()
        
        # Interface selection
        toolbar.addWidget(QLabel("Interface:"))
        self.interface_combo = QComboBox()
        self.interface_combo.addItems(["Auto", "wlan0", "wlan1", "Wi-Fi"])
        toolbar.addWidget(self.interface_combo)
    
    def setup_connections(self):
        """Setup signal connections"""
        pass  # Connections are set up in individual methods
    
    def start_realtime_capture(self):
        """Start real-time WiFi signal capture"""
        if self.signal_thread and self.signal_thread.isRunning():
            return
        
        interface = self.interface_combo.currentText()
        if interface == "Auto":
            interface = None
        
        # Create and start signal thread
        self.signal_thread = RealTimeSignalThread()
        self.signal_thread.signal_updated.connect(self.on_signals_updated)
        self.signal_thread.image_generated.connect(self.on_image_generated)
        self.signal_thread.ai_processing_complete.connect(self.on_ai_complete)
        self.signal_thread.error_occurred.connect(self.on_capture_error)
        
        self.signal_thread.start_capture(interface)
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_bar.showMessage(f"Capturing WiFi signals on interface: {interface or 'auto'}")
        
    def stop_realtime_capture(self):
        """Stop real-time WiFi signal capture"""
        if self.signal_thread:
            self.signal_thread.stop_capture()
            self.signal_thread = None
        
        # Update UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage("WiFi signal capture stopped")
    
    def on_signals_updated(self, signals: List):
        """Handle updated signal data"""
        self.analysis_panel.update_stats(signals)
        self.status_bar.showMessage(f"Signals: {len(signals)} | Last update: {datetime.now().strftime('%H:%M:%S')}")
    
    def on_image_generated(self, image: np.ndarray, image_type: str):
        """Handle generated visualization image"""
        self.visualization_widget.update_visualization(image, image_type)
    
    def on_ai_complete(self, results: Dict):
        """Handle AI processing completion"""
        self.analysis_panel.update_ai_analysis(results)
        
        # Show notification
        self.status_bar.showMessage("AI analysis complete", 3000)
        
        # Load AI-generated images if available
        output_files = results.get("output_files", [])
        for file_path in output_files:
            if file_path.endswith("_ai_generated.png"):
                ai_image = cv2.imread(file_path)
                if ai_image is not None:
                    self.visualization_widget.update_visualization(ai_image, "AI Generated")
                break
    
    def on_capture_error(self, error_message: str):
        """Handle capture errors"""
        QMessageBox.critical(self, "Capture Error", error_message)
        self.stop_realtime_capture()
    
    def process_with_ai(self):
        """Process current signals with AI"""
        if not self.signal_thread or not self.signal_thread.capture_system:
            QMessageBox.information(self, "No Data", "Start signal capture first")
            return
        
        signals = self.signal_thread.capture_system.get_current_signals()
        if not signals:
            QMessageBox.information(self, "No Signals", "No signal data available for AI processing")
            return
        
        # Show processing dialog
        progress = QProgressDialog("Processing with AI models...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            # Process with AI in separate thread
            ai_converter = WiFiSignalToImageAI()
            
            def ai_processing():
                results = ai_converter.process_wifi_signals(
                    signals[-200:],  # Last 200 points
                    f"manual_{int(time.time())}"
                )
                self.on_ai_complete(results)
                progress.close()
            
            ai_thread = threading.Thread(target=ai_processing, daemon=True)
            ai_thread.start()
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "AI Processing Error", str(e))
    
    def load_signal_data(self):
        """Load signal data from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Signal Data",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Process loaded data
                QMessageBox.information(self, "Loaded", f"Signal data loaded from {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load data: {str(e)}")
    
    def save_session(self):
        """Save current session data"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            f"wifi_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                session_data = {
                    "timestamp": time.time(),
                    "interface": self.interface_combo.currentText(),
                    "capture_active": self.signal_thread is not None and self.signal_thread.isRunning()
                }
                
                if self.signal_thread and self.signal_thread.capture_system:
                    signals = self.signal_thread.capture_system.get_current_signals()
                    session_data["signals"] = [
                        {
                            "timestamp": s.timestamp,
                            "frequency": s.frequency,
                            "signal_strength": s.signal_strength,
                            "ssid": s.ssid,
                            "channel": s.channel
                        }
                        for s in signals
                    ]
                
                with open(filename, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                QMessageBox.information(self, "Saved", f"Session saved to {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save session: {str(e)}")
    
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        dark_style = """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QTabWidget::pane {
            border: 1px solid #555;
        }
        QTabBar::tab {
            background-color: #404040;
            border: 1px solid #555;
            padding: 8px;
        }
        QTabBar::tab:selected {
            background-color: #606060;
        }
        QPushButton {
            background-color: #404040;
            border: 1px solid #555;
            padding: 8px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #505050;
        }
        QPushButton:pressed {
            background-color: #606060;
        }
        QLabel {
            color: #ffffff;
        }
        QTableWidget {
            background-color: #353535;
            alternate-background-color: #404040;
            gridline-color: #555;
        }
        QHeaderView::section {
            background-color: #404040;
            border: 1px solid #555;
            padding: 4px;
        }
        """
        self.setStyleSheet(dark_style)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>WiFi Vision Pro</h2>
        <p><b>Advanced WiFi Signal Visualization with AI Integration</b></p>
        <p>Version 2.0.0 - Multi-Platform Edition</p>
        
        <h3>Features:</h3>
        <ul>
        <li>🔄 Real-time WiFi signal capture and visualization</li>
        <li>🤖 AI-powered signal-to-image conversion using Hugging Face models</li>
        <li>📊 Advanced signal analysis and interference detection</li>
        <li>🎨 Multiple visualization modes (heatmap, spectrogram, AI-generated)</li>
        <li>📈 Real-time charts and statistical analysis</li>
        <li>🌐 Cross-platform support (Windows, Linux, macOS)</li>
        <li>💾 Session save/load functionality</li>
        </ul>
        
        <h3>AI Models Used:</h3>
        <ul>
        <li>Vision Transformer (ViT) for image analysis</li>
        <li>Stable Diffusion for AI image generation</li>
        <li>Audio processing models for signal interpretation</li>
        </ul>
        
        <p><b>Developed with:</b> Python, Qt, OpenCV, Hugging Face Transformers, PyTorch</p>
        <p><b>License:</b> Commercial - WiFi Analysis Solutions</p>
        """
        
        QMessageBox.about(self, "About WiFi Vision Pro", about_text)
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.signal_thread and self.signal_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Exit Confirmation",
                "WiFi capture is still running. Stop capture and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.stop_realtime_capture()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("WiFi Vision Pro")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("WiFi Analysis Solutions")
    
    # Create and show main window
    window = WiFiVisionProMainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()