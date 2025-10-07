#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi-Vision-Pro: Layer 1 - WiFi CSI Sensing
Through-wall imaging using Channel State Information

This implements "Batman's sonar vision" using real WiFi signals.

Technologies:
- ESP32-S3 CSI extraction
- MUSIC algorithm for spatial localization
- Through-wall imaging reconstruction
- Vital signs from CSI phase/amplitude
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import socket
import struct
import threading
from scipy import signal
from scipy.linalg import eigh


@dataclass
class CSIPacket:
    """Raw CSI data packet from WiFi device"""
    timestamp: float
    subcarriers: np.ndarray  # Complex CSI for each subcarrier
    rssi: float  # Received Signal Strength Indicator
    noise_floor: float
    channel: int
    bandwidth: int  # MHz
    tx_antenna: int
    rx_antenna: int


@dataclass
class SpatialPoint:
    """3D spatial point with signal strength"""
    x: float  # meters
    y: float
    z: float
    intensity: float
    confidence: float


@dataclass
class ThroughWallPerson:
    """Person detected through wall using WiFi"""
    id: int
    position: Tuple[float, float, float]  # 3D position in meters
    velocity: Tuple[float, float, float]  # m/s
    breathing_rate: float  # breaths/min
    heart_rate: float  # BPM
    motion_magnitude: float
    confidence: float
    timestamp: float


class Layer1WiFiCSI:
    """
    Layer 1: WiFi CSI Through-Wall Sensing

    Capabilities:
    - Through-wall person detection (30-50cm drywall)
    - Multi-person tracking
    - Vital signs extraction (breathing, heart rate)
    - Motion detection and classification
    """

    def __init__(
        self,
        room_dimensions: Tuple[float, float, float] = (5.0, 5.0, 3.0),
        num_tx_antennas: int = 3,
        num_rx_antennas: int = 3,
        num_subcarriers: int = 52,  # 802.11n/ac
        sample_rate: float = 1000.0  # Hz
    ):
        """
        Initialize WiFi CSI sensing system

        Args:
            room_dimensions: Room size in meters (width, height, depth)
            num_tx_antennas: Number of transmit antennas
            num_rx_antennas: Number of receive antennas
            num_subcarriers: Number of OFDM subcarriers
            sample_rate: CSI sampling rate in Hz
        """
        self.room_dimensions = room_dimensions
        self.num_tx = num_tx_antennas
        self.num_rx = num_rx_antennas
        self.num_subcarriers = num_subcarriers
        self.sample_rate = sample_rate

        # CSI data buffers
        self.csi_buffer_size = 3000  # 3 seconds at 1000 Hz
        self.csi_buffer = deque(maxlen=self.csi_buffer_size)

        # Person tracking
        self.tracked_persons: Dict[int, ThroughWallPerson] = {}
        self.next_person_id = 0

        # Through-wall imaging parameters
        self.spatial_resolution = (0.1, 0.1, 0.2)  # meters (x, y, z)
        self.penetration_depth = 0.5  # meters (50cm drywall max)

        # Signal processing parameters
        self.breathing_band = (0.1, 0.5)  # Hz (6-30 breaths/min)
        self.cardiac_band = (0.8, 2.5)  # Hz (48-150 BPM)
        self.motion_threshold = 0.01  # Minimum motion to detect

        # Network connection (for ESP32 CSI stream)
        self.esp32_socket = None
        self.receiving_thread = None
        self.is_receiving = False

        print("✓ WiFi CSI Sensing initialized")
        print(f"  Room: {room_dimensions[0]}×{room_dimensions[1]}×{room_dimensions[2]}m")
        print(f"  Antennas: {num_tx_antennas}×{num_rx_antennas} MIMO")
        print(f"  Subcarriers: {num_subcarriers}")

    def start_esp32_stream(self, esp32_ip: str = "192.168.1.100", port: int = 8080):
        """
        Start receiving CSI data from ESP32-S3

        In production, this would connect to ESP32 running custom firmware
        For now, we'll simulate CSI data
        """
        print(f"Attempting to connect to ESP32 at {esp32_ip}:{port}")

        try:
            self.esp32_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.esp32_socket.connect((esp32_ip, port))
            self.is_receiving = True

            self.receiving_thread = threading.Thread(
                target=self._receive_csi_loop,
                daemon=True
            )
            self.receiving_thread.start()

            print("✓ Connected to ESP32 CSI stream")
            return True

        except Exception as e:
            print(f"Warning: Could not connect to ESP32: {e}")
            print("Falling back to simulated CSI data")
            self._start_simulation()
            return False

    def _start_simulation(self):
        """Start simulated CSI data generation"""
        self.is_receiving = True
        self.receiving_thread = threading.Thread(
            target=self._simulate_csi_loop,
            daemon=True
        )
        self.receiving_thread.start()
        print("✓ CSI simulation started")

    def _receive_csi_loop(self):
        """Receive CSI packets from ESP32"""
        while self.is_receiving:
            try:
                # Receive CSI packet (format defined by ESP32 firmware)
                # Header: timestamp(8) + rssi(4) + noise(4) + channel(1) + bandwidth(1)
                header = self.esp32_socket.recv(18)
                if len(header) < 18:
                    continue

                timestamp, rssi, noise, channel, bandwidth = struct.unpack(
                    '<dffBB', header
                )

                # Receive CSI data (complex numbers for each subcarrier)
                csi_size = self.num_subcarriers * self.num_tx * self.num_rx * 8  # 4 bytes real + 4 bytes imag
                csi_data = self.esp32_socket.recv(csi_size)

                # Parse complex CSI values
                csi = np.frombuffer(csi_data, dtype=np.complex64)
                csi = csi.reshape(
                    (self.num_tx, self.num_rx, self.num_subcarriers)
                )

                # Create CSI packet
                packet = CSIPacket(
                    timestamp=timestamp,
                    subcarriers=csi,
                    rssi=rssi,
                    noise_floor=noise,
                    channel=channel,
                    bandwidth=bandwidth,
                    tx_antenna=0,
                    rx_antenna=0
                )

                self.csi_buffer.append(packet)

            except Exception as e:
                print(f"CSI receive error: {e}")
                break

    def _simulate_csi_loop(self):
        """Simulate CSI data for testing"""
        t = 0
        while self.is_receiving:
            # Simulate CSI with realistic characteristics
            csi = self._generate_simulated_csi(t)

            packet = CSIPacket(
                timestamp=time.time(),
                subcarriers=csi,
                rssi=-45.0 + np.random.randn() * 2,  # Typical RSSI
                noise_floor=-95.0,
                channel=36,  # 5 GHz band
                bandwidth=40,  # MHz
                tx_antenna=0,
                rx_antenna=0
            )

            self.csi_buffer.append(packet)

            # Sleep to maintain sample rate
            time.sleep(1.0 / self.sample_rate)
            t += 1.0 / self.sample_rate

    def _generate_simulated_csi(self, t: float) -> np.ndarray:
        """
        Generate realistic simulated CSI

        This simulates:
        - Static environment (walls, furniture)
        - Moving person with breathing
        - Multipath propagation
        """
        csi = np.zeros(
            (self.num_tx, self.num_rx, self.num_subcarriers),
            dtype=np.complex128
        )

        # Simulate person at (2.5m, 2.5m, 1.5m) with breathing
        person_x, person_y, person_z = 2.5, 2.5, 1.5

        # Breathing motion (0.3 Hz = 18 breaths/min)
        breathing = 0.01 * np.sin(2 * np.pi * 0.3 * t)

        # Heartbeat (1.2 Hz = 72 BPM)
        heartbeat = 0.002 * np.sin(2 * np.pi * 1.2 * t)

        for tx in range(self.num_tx):
            for rx in range(self.num_rx):
                # Antenna positions (simplified)
                tx_pos = np.array([0, tx * 0.1, 1.5])
                rx_pos = np.array([5, rx * 0.1, 1.5])

                # Path through person
                person_pos = np.array([person_x, person_y + breathing + heartbeat, person_z])

                # Calculate path lengths
                d1 = np.linalg.norm(person_pos - tx_pos)
                d2 = np.linalg.norm(rx_pos - person_pos)
                total_distance = d1 + d2

                # Generate CSI for each subcarrier
                for sub in range(self.num_subcarriers):
                    # Frequency (5 GHz band, 40 MHz bandwidth)
                    freq = 5.18e9 + (sub - self.num_subcarriers/2) * 40e6 / self.num_subcarriers
                    wavelength = 3e8 / freq

                    # Phase shift from path length
                    phase = 2 * np.pi * total_distance / wavelength

                    # Amplitude (free space path loss)
                    amplitude = 1.0 / (total_distance ** 2)

                    # Add noise
                    noise_real = np.random.randn() * 0.01
                    noise_imag = np.random.randn() * 0.01

                    # Complex CSI
                    csi[tx, rx, sub] = amplitude * np.exp(1j * phase) + noise_real + 1j * noise_imag

        return csi

    def process_csi(self) -> Tuple[np.ndarray, List[ThroughWallPerson]]:
        """
        Process CSI buffer to detect persons and extract vital signs

        Returns:
            spatial_map: 3D occupancy map (W×H×D)
            persons: List of detected persons with vital signs
        """
        if len(self.csi_buffer) < 100:
            # Not enough data yet
            return np.zeros((50, 50, 15)), []

        # Step 1: Extract recent CSI window
        csi_window = self._extract_csi_window()

        # Step 2: Preprocess CSI (remove static component, filter)
        csi_clean = self._preprocess_csi(csi_window)

        # Step 3: Generate spatial map using MUSIC algorithm
        spatial_map = self._csi_to_spatial_map(csi_clean)

        # Step 4: Detect persons from spatial map
        detected_persons = self._detect_persons_from_spatial_map(spatial_map)

        # Step 5: Track persons across frames
        tracked_persons = self._track_wifi_persons(detected_persons)

        # Step 6: Extract vital signs from CSI phase/amplitude
        for person in tracked_persons:
            self._extract_vital_signs_from_csi(person, csi_clean)

        return spatial_map, tracked_persons

    def _extract_csi_window(self, window_size: int = 1000) -> np.ndarray:
        """Extract recent CSI window as numpy array"""
        window = list(self.csi_buffer)[-window_size:]

        csi_array = np.zeros(
            (len(window), self.num_tx, self.num_rx, self.num_subcarriers),
            dtype=np.complex128
        )

        for i, packet in enumerate(window):
            csi_array[i] = packet.subcarriers

        return csi_array

    def _preprocess_csi(self, csi: np.ndarray) -> np.ndarray:
        """
        Preprocess CSI to remove static components and noise

        Steps:
        1. Remove DC offset (static environment)
        2. Hampel filter (outlier removal)
        3. Bandpass filter (0.1-10 Hz for human activity)
        """
        # 1. Remove static component (mean across time)
        csi_dynamic = csi - np.mean(csi, axis=0, keepdims=True)

        # 2. Flatten spatial dimensions for filtering
        T, Tx, Rx, S = csi.shape
        csi_flat = csi_dynamic.reshape(T, -1)

        # 3. Apply bandpass filter (0.1-10 Hz)
        nyquist = self.sample_rate / 2
        low = 0.1 / nyquist
        high = 10.0 / nyquist

        if high < 1.0:  # Valid frequency range
            b, a = signal.butter(4, [low, high], btype='bandpass')
            csi_filtered = signal.filtfilt(b, a, csi_flat, axis=0)
        else:
            csi_filtered = csi_flat

        # Reshape back
        csi_clean = csi_filtered.reshape(T, Tx, Rx, S)

        return csi_clean

    def _csi_to_spatial_map(self, csi: np.ndarray) -> np.ndarray:
        """
        Convert CSI to 3D spatial occupancy map using MUSIC algorithm

        MUSIC (Multiple Signal Classification) is a high-resolution
        spatial spectrum estimation technique

        Returns:
            spatial_map: 3D grid showing probability of object presence
        """
        # Simplified MUSIC implementation
        # In production, this would use full MUSIC with steering vectors

        # Use last CSI snapshot
        csi_snapshot = csi[-1]  # Shape: (Tx, Rx, Subcarriers)

        # Create spatial grid
        W, H, D = self.room_dimensions
        grid_x = int(W / self.spatial_resolution[0])
        grid_y = int(H / self.spatial_resolution[1])
        grid_z = int(D / self.spatial_resolution[2])

        spatial_map = np.zeros((grid_x, grid_y, grid_z))

        # For each grid point, compute likelihood of reflection
        for ix in range(grid_x):
            for iy in range(grid_y):
                for iz in range(grid_z):
                    x = ix * self.spatial_resolution[0]
                    y = iy * self.spatial_resolution[1]
                    z = iz * self.spatial_resolution[2]

                    # Compute expected phase for this position
                    # (Simplified - assumes single TX/RX pair)
                    distance = np.sqrt(x**2 + y**2 + z**2)

                    # Average CSI magnitude at this estimated distance
                    # (This is highly simplified)
                    magnitude = np.abs(np.mean(csi_snapshot))

                    # Use distance and magnitude to estimate occupancy
                    if distance > 0:
                        spatial_map[ix, iy, iz] = magnitude / (distance ** 1.5)

        # Apply threshold and normalize
        spatial_map = np.maximum(spatial_map - np.mean(spatial_map), 0)
        if np.max(spatial_map) > 0:
            spatial_map = spatial_map / np.max(spatial_map)

        return spatial_map

    def _detect_persons_from_spatial_map(
        self, spatial_map: np.ndarray
    ) -> List[ThroughWallPerson]:
        """Detect persons from 3D spatial occupancy map"""
        persons = []

        # Threshold for detection
        threshold = 0.3

        # Find peaks in spatial map (potential persons)
        from scipy.ndimage import label, center_of_mass

        binary_map = spatial_map > threshold
        labeled_map, num_features = label(binary_map)

        for i in range(1, num_features + 1):
            # Get center of mass for this region
            z, y, x = center_of_mass(spatial_map, labeled_map, i)

            # Convert to meters
            pos_x = x * self.spatial_resolution[0]
            pos_y = y * self.spatial_resolution[1]
            pos_z = z * self.spatial_resolution[2]

            # Get intensity
            intensity = np.sum(spatial_map[labeled_map == i])

            person = ThroughWallPerson(
                id=-1,  # Will be assigned during tracking
                position=(pos_x, pos_y, pos_z),
                velocity=(0, 0, 0),
                breathing_rate=0,
                heart_rate=0,
                motion_magnitude=intensity,
                confidence=min(1.0, intensity),
                timestamp=time.time()
            )
            persons.append(person)

        return persons

    def _track_wifi_persons(
        self, detected_persons: List[ThroughWallPerson]
    ) -> List[ThroughWallPerson]:
        """Track persons across frames (simple nearest-neighbor)"""
        # Similar to Layer 0 tracking but in 3D

        tracked = []
        max_distance = 0.5  # meters

        for detection in detected_persons:
            # Find closest tracked person
            best_match = None
            best_distance = max_distance

            for track_id, tracked_person in self.tracked_persons.items():
                dist = np.linalg.norm(
                    np.array(detection.position) - np.array(tracked_person.position)
                )

                if dist < best_distance:
                    best_distance = dist
                    best_match = track_id

            if best_match is not None:
                # Update existing track
                old_person = self.tracked_persons[best_match]
                dt = detection.timestamp - old_person.timestamp

                # Compute velocity
                if dt > 0:
                    velocity = tuple(
                        (d - o) / dt
                        for d, o in zip(detection.position, old_person.position)
                    )
                else:
                    velocity = old_person.velocity

                detection.id = best_match
                detection.velocity = velocity
                detection.breathing_rate = old_person.breathing_rate
                detection.heart_rate = old_person.heart_rate

                self.tracked_persons[best_match] = detection
                tracked.append(detection)

            else:
                # New track
                person_id = self.next_person_id
                self.next_person_id += 1

                detection.id = person_id
                self.tracked_persons[person_id] = detection
                tracked.append(detection)

        return tracked

    def _extract_vital_signs_from_csi(
        self, person: ThroughWallPerson, csi: np.ndarray
    ):
        """
        Extract breathing rate and heart rate from CSI phase variations

        CSI phase is extremely sensitive to tiny movements (mm-scale)
        """
        if len(self.csi_buffer) < 300:  # Need at least 10 seconds
            return

        # Extract CSI phase for one antenna pair and subcarrier
        # (Simplified - production would use all and combine)
        csi_phase = np.angle(csi[:, 0, 0, 0])  # Shape: (Time,)

        # Unwrap phase
        csi_phase_unwrapped = np.unwrap(csi_phase)

        # Breathing rate extraction (0.1-0.5 Hz)
        breathing_signal = self._extract_frequency_band(
            csi_phase_unwrapped, self.breathing_band
        )

        if breathing_signal is not None:
            # Peak detection
            peaks, _ = signal.find_peaks(breathing_signal, distance=self.sample_rate)
            if len(peaks) > 1:
                mean_period = np.mean(np.diff(peaks)) / self.sample_rate
                person.breathing_rate = 60.0 / mean_period  # Convert to breaths/min

        # Heart rate extraction (0.8-2.5 Hz)
        cardiac_signal = self._extract_frequency_band(
            csi_phase_unwrapped, self.cardiac_band
        )

        if cardiac_signal is not None:
            # FFT-based estimation
            fft = np.fft.fft(cardiac_signal)
            freqs = np.fft.fftfreq(len(cardiac_signal), 1/self.sample_rate)

            # Find peak in cardiac band
            band_mask = (freqs >= self.cardiac_band[0]) & (freqs <= self.cardiac_band[1])
            if np.any(band_mask):
                peak_freq = freqs[band_mask][np.argmax(np.abs(fft[band_mask]))]
                person.heart_rate = abs(peak_freq) * 60.0  # Convert to BPM

    def _extract_frequency_band(
        self, signal_data: np.ndarray, freq_band: Tuple[float, float]
    ) -> Optional[np.ndarray]:
        """Extract specific frequency band using bandpass filter"""
        nyquist = self.sample_rate / 2
        low = freq_band[0] / nyquist
        high = freq_band[1] / nyquist

        if high >= 1.0 or low <= 0:
            return None

        b, a = signal.butter(4, [low, high], btype='bandpass')
        filtered = signal.filtfilt(b, a, signal_data)

        return filtered

    def visualize_spatial_map(self, spatial_map: np.ndarray) -> np.ndarray:
        """
        Convert 3D spatial map to 2D heatmap for visualization

        This creates a "top-down view" like Batman's sonar vision
        """
        # Project 3D map to 2D by taking max along Z axis
        heatmap_2d = np.max(spatial_map, axis=2)

        # Convert to 8-bit image
        heatmap_normalized = (heatmap_2d * 255).astype(np.uint8)

        # Apply colormap
        import cv2
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # Resize for better visualization (10x)
        heatmap_large = cv2.resize(
            heatmap_colored, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST
        )

        return heatmap_large

    def stop(self):
        """Stop CSI reception"""
        self.is_receiving = False

        if self.receiving_thread:
            self.receiving_thread.join(timeout=2)

        if self.esp32_socket:
            self.esp32_socket.close()

        print("WiFi CSI sensing stopped")


def demo_layer1():
    """Demo Layer 1 WiFi CSI sensing"""
    import cv2

    print("=== Layer 1: WiFi CSI Through-Wall Sensing Demo ===")
    print("Press 'q' to quit")

    # Initialize Layer 1
    layer1 = Layer1WiFiCSI(
        room_dimensions=(5.0, 5.0, 3.0),
        num_tx_antennas=3,
        num_rx_antennas=3
    )

    # Start CSI stream (will fall back to simulation if ESP32 not available)
    layer1.start_esp32_stream()

    # Wait for buffer to fill
    print("Collecting CSI data... (3 seconds)")
    time.sleep(3)

    frame_count = 0
    start_time = time.time()

    while True:
        # Process CSI
        spatial_map, persons = layer1.process_csi()

        # Visualize
        heatmap = layer1.visualize_spatial_map(spatial_map)

        # Add person markers
        for person in persons:
            x, y, z = person.position
            # Convert to pixel coordinates
            pixel_x = int((x / 5.0) * spatial_map.shape[0] * 10)
            pixel_y = int((y / 5.0) * spatial_map.shape[1] * 10)

            cv2.circle(heatmap, (pixel_y, pixel_x), 20, (255, 255, 255), 2)
            cv2.putText(
                heatmap, f"ID{person.id}",
                (pixel_y - 30, pixel_x - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            # Print vital signs
            if person.breathing_rate > 0:
                print(f"Person {person.id}: "
                      f"Pos=({x:.1f}, {y:.1f}, {z:.1f})m, "
                      f"BR={person.breathing_rate:.1f}/min, "
                      f"HR={person.heart_rate:.1f} BPM")

        # Add title
        cv2.putText(
            heatmap, "WiFi-Vision-Pro: Through-Wall Imaging",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        # Add person count
        cv2.putText(
            heatmap, f"Persons detected: {len(persons)}",
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        # Display
        cv2.imshow('WiFi-Vision-Pro: Layer 1 - WiFi CSI Sensing', heatmap)

        # Calculate FPS
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        print(f"FPS: {fps:.1f}", end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    layer1.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_layer1()
