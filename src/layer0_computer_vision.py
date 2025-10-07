#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi-Vision-Pro: Layer 0 - Computer Vision
Professional-grade real-time person detection and vital signs estimation

Technologies:
- YOLOv8 for person detection
- MediaPipe for pose estimation
- OpenCV for video processing
- Vital signs estimation from video (rPPG)
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import mediapipe as mp

# Try to import YOLOv8 (ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")


@dataclass
class Person:
    """Detected person with tracking information"""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    center: Tuple[int, int]
    keypoints: Optional[np.ndarray] = None  # MediaPipe landmarks
    vital_signs: Optional[Dict[str, float]] = None
    track_history: deque = None  # Movement history

    def __post_init__(self):
        if self.track_history is None:
            self.track_history = deque(maxlen=30)  # Last 30 frames


@dataclass
class VitalSigns:
    """Vital signs measurements"""
    heart_rate: float  # BPM
    respiratory_rate: float  # breaths/min
    heart_rate_variability: float  # RMSSD in ms
    confidence: float  # 0-1
    timestamp: float


class Layer0ComputerVision:
    """
    Layer 0: Standard Computer Vision
    - Person detection and tracking
    - Pose estimation
    - Vital signs estimation from video (remote photoplethysmography)
    """

    def __init__(self, use_gpu: bool = True):
        """Initialize Layer 0 computer vision system"""
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0

        # Initialize YOLOv8 for person detection
        self.detector = None
        if YOLO_AVAILABLE:
            try:
                self.detector = YOLO('yolov8n.pt')  # Nano model for speed
                if self.use_gpu:
                    self.detector.to('cuda')
                print("✓ YOLOv8 initialized")
            except Exception as e:
                print(f"Warning: YOLOv8 initialization failed: {e}")

        # Initialize MediaPipe for pose estimation
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_estimator = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe Pose initialized")

        # Person tracking
        self.tracked_persons: Dict[int, Person] = {}
        self.next_person_id = 0
        self.max_track_distance = 100  # pixels

        # Vital signs estimation buffers
        self.vital_signs_buffer_size = 300  # 10 seconds at 30 FPS
        self.roi_color_buffer: Dict[int, deque] = {}  # Per-person color buffer

        # Performance monitoring
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Person]]:
        """
        Process single video frame

        Args:
            frame: Input BGR frame from camera

        Returns:
            annotated_frame: Frame with visualizations
            persons: List of detected persons with vital signs
        """
        self.frame_count += 1

        # Step 1: Detect persons using YOLOv8
        detected_persons = self._detect_persons(frame)

        # Step 2: Track persons across frames
        tracked_persons = self._track_persons(detected_persons)

        # Step 3: Estimate pose for each person
        for person in tracked_persons:
            self._estimate_pose(frame, person)

        # Step 4: Estimate vital signs from video
        for person in tracked_persons:
            self._estimate_vital_signs(frame, person)

        # Step 5: Annotate frame with detections and vital signs
        annotated_frame = self._annotate_frame(frame.copy(), tracked_persons)

        # Update FPS
        self._update_fps()

        return annotated_frame, tracked_persons

    def _detect_persons(self, frame: np.ndarray) -> List[Person]:
        """Detect persons using YOLOv8"""
        persons = []

        if self.detector is None:
            # Fallback: Use Haar Cascade (less accurate but no dependencies)
            return self._detect_persons_cascade(frame)

        # Run YOLOv8 detection
        results = self.detector(frame, classes=[0], verbose=False)  # class 0 = person

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])

                # Convert to x, y, w, h format
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                center = (x + w // 2, y + h // 2)

                person = Person(
                    id=-1,  # Will be assigned during tracking
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    center=center
                )
                persons.append(person)

        return persons

    def _detect_persons_cascade(self, frame: np.ndarray) -> List[Person]:
        """Fallback person detection using Haar Cascade"""
        # This is a simple fallback - in production you'd want better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use upper body detector as fallback
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_upperbody.xml'
        )

        detections = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        persons = []
        for (x, y, w, h) in detections:
            center = (x + w // 2, y + h // 2)
            person = Person(
                id=-1,
                bbox=(x, y, w, h),
                confidence=0.7,  # Fixed confidence for cascade
                center=center
            )
            persons.append(person)

        return persons

    def _track_persons(self, detected_persons: List[Person]) -> List[Person]:
        """Track persons across frames using simple IOU tracking"""
        tracked = []

        # Match detected persons to tracked persons
        unmatched_detections = set(range(len(detected_persons)))
        unmatched_tracks = set(self.tracked_persons.keys())

        matches = []

        # Compute distance matrix between detections and tracks
        for det_idx, detection in enumerate(detected_persons):
            best_match = None
            best_distance = self.max_track_distance

            for track_id in unmatched_tracks:
                tracked_person = self.tracked_persons[track_id]
                distance = self._compute_distance(
                    detection.center, tracked_person.center
                )

                if distance < best_distance:
                    best_distance = distance
                    best_match = track_id

            if best_match is not None:
                matches.append((det_idx, best_match))
                unmatched_detections.discard(det_idx)
                unmatched_tracks.discard(best_match)

        # Update matched tracks
        for det_idx, track_id in matches:
            detection = detected_persons[det_idx]
            tracked_person = self.tracked_persons[track_id]

            # Update tracked person
            tracked_person.bbox = detection.bbox
            tracked_person.center = detection.center
            tracked_person.confidence = detection.confidence
            tracked_person.track_history.append(detection.center)

            tracked.append(tracked_person)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detected_persons[det_idx]
            person_id = self.next_person_id
            self.next_person_id += 1

            detection.id = person_id
            detection.track_history = deque(maxlen=30)
            detection.track_history.append(detection.center)

            self.tracked_persons[person_id] = detection
            self.roi_color_buffer[person_id] = deque(maxlen=self.vital_signs_buffer_size)

            tracked.append(detection)

        # Remove lost tracks
        for track_id in unmatched_tracks:
            del self.tracked_persons[track_id]
            if track_id in self.roi_color_buffer:
                del self.roi_color_buffer[track_id]

        return tracked

    def _compute_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Compute Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _estimate_pose(self, frame: np.ndarray, person: Person):
        """Estimate pose using MediaPipe"""
        x, y, w, h = person.bbox

        # Extract person ROI
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return

        # Convert to RGB for MediaPipe
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Run pose estimation
        results = self.pose_estimator.process(roi_rgb)

        if results.pose_landmarks:
            # Store landmarks (33 keypoints)
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                # Convert normalized coordinates to frame coordinates
                lm_x = int(landmark.x * w + x)
                lm_y = int(landmark.y * h + y)
                landmarks.append((lm_x, lm_y, landmark.visibility))

            person.keypoints = np.array(landmarks)

    def _estimate_vital_signs(self, frame: np.ndarray, person: Person):
        """
        Estimate vital signs using remote photoplethysmography (rPPG)

        This is a simplified version. Production would use:
        - Face detection for better ROI
        - ICA/PCA for signal separation
        - Bandpass filtering
        - Peak detection algorithms
        """
        x, y, w, h = person.bbox

        # Extract forehead/face region (upper 30% of bbox)
        face_roi_y = y + int(h * 0.1)
        face_roi_h = int(h * 0.3)
        face_roi = frame[face_roi_y:face_roi_y+face_roi_h, x:x+w]

        if face_roi.size == 0:
            return

        # Compute mean color in green channel (most sensitive to blood flow)
        mean_green = np.mean(face_roi[:, :, 1])

        # Store in buffer
        if person.id in self.roi_color_buffer:
            self.roi_color_buffer[person.id].append(mean_green)

        # Compute vital signs if we have enough data
        buffer = self.roi_color_buffer.get(person.id, deque())
        if len(buffer) >= 150:  # At least 5 seconds at 30 FPS
            vital_signs = self._compute_vital_signs_from_signal(
                np.array(buffer), fps=30
            )
            person.vital_signs = vital_signs

    def _compute_vital_signs_from_signal(
        self, signal: np.ndarray, fps: float = 30
    ) -> Dict[str, float]:
        """
        Compute heart rate and respiratory rate from rPPG signal

        This is simplified. Production version would use:
        - Butterworth bandpass filter (0.7-4 Hz for HR, 0.1-0.5 Hz for RR)
        - FFT for frequency analysis
        - Peak detection for HRV
        """
        # Detrend signal
        signal_detrended = signal - np.mean(signal)

        # Simple FFT-based heart rate estimation
        fft = np.fft.fft(signal_detrended)
        freqs = np.fft.fftfreq(len(signal), 1/fps)

        # Heart rate band: 0.7-4 Hz (42-240 BPM)
        hr_band = (freqs >= 0.7) & (freqs <= 4.0)
        hr_fft = np.abs(fft[hr_band])
        hr_freqs = freqs[hr_band]

        if len(hr_fft) > 0:
            peak_freq = hr_freqs[np.argmax(hr_fft)]
            heart_rate = peak_freq * 60  # Convert to BPM
        else:
            heart_rate = 0

        # Respiratory rate band: 0.1-0.5 Hz (6-30 breaths/min)
        rr_band = (freqs >= 0.1) & (freqs <= 0.5)
        rr_fft = np.abs(fft[rr_band])
        rr_freqs = freqs[rr_band]

        if len(rr_fft) > 0:
            peak_freq_rr = rr_freqs[np.argmax(rr_fft)]
            respiratory_rate = peak_freq_rr * 60  # Convert to breaths/min
        else:
            respiratory_rate = 0

        # Compute HRV (simplified - just signal variance)
        hrv = np.std(signal_detrended)

        # Confidence based on signal quality
        snr = np.max(hr_fft) / np.mean(hr_fft) if len(hr_fft) > 0 else 0
        confidence = min(1.0, snr / 10.0)

        return {
            'heart_rate': float(heart_rate),
            'respiratory_rate': float(respiratory_rate),
            'hrv': float(hrv),
            'confidence': float(confidence)
        }

    def _annotate_frame(
        self, frame: np.ndarray, persons: List[Person]
    ) -> np.ndarray:
        """Annotate frame with detections, poses, and vital signs"""

        for person in persons:
            x, y, w, h = person.bbox

            # Draw bounding box
            color = (0, 255, 0) if person.confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Draw person ID
            cv2.putText(
                frame, f"ID: {person.id}",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

            # Draw pose keypoints if available
            if person.keypoints is not None:
                self._draw_pose(frame, person.keypoints)

            # Draw track history (movement trail)
            if len(person.track_history) > 1:
                points = np.array(list(person.track_history), dtype=np.int32)
                cv2.polylines(frame, [points], False, (255, 0, 255), 2)

            # Draw vital signs if available
            if person.vital_signs:
                self._draw_vital_signs(frame, person)

        # Draw FPS
        cv2.putText(
            frame, f"FPS: {self.fps:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )

        # Draw person count
        cv2.putText(
            frame, f"Persons: {len(persons)}",
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )

        return frame

    def _draw_pose(self, frame: np.ndarray, keypoints: np.ndarray):
        """Draw pose skeleton"""
        # MediaPipe pose connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
        ]

        # Draw skeleton
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))

                # Only draw if both points are visible
                if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
                    cv2.line(frame, start_point, end_point, (255, 255, 0), 2)

        # Draw keypoints
        for kp in keypoints:
            if kp[2] > 0.5:  # Visibility threshold
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 255, 255), -1)

    def _draw_vital_signs(self, frame: np.ndarray, person: Person):
        """Draw vital signs overlay"""
        vs = person.vital_signs
        x, y, w, h = person.bbox

        # Create semi-transparent overlay
        overlay_y = y + h + 10
        overlay_height = 120

        # Background rectangle
        cv2.rectangle(
            frame,
            (x, overlay_y),
            (x + max(w, 300), overlay_y + overlay_height),
            (0, 0, 0), -1
        )
        cv2.rectangle(
            frame,
            (x, overlay_y),
            (x + max(w, 300), overlay_y + overlay_height),
            (255, 255, 255), 2
        )

        # Heart rate
        hr_color = (0, 255, 0) if 60 <= vs['heart_rate'] <= 100 else (0, 0, 255)
        cv2.putText(
            frame, f"HR: {vs['heart_rate']:.1f} BPM",
            (x + 10, overlay_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, hr_color, 2
        )

        # Respiratory rate
        rr_color = (0, 255, 0) if 12 <= vs['respiratory_rate'] <= 20 else (0, 0, 255)
        cv2.putText(
            frame, f"RR: {vs['respiratory_rate']:.1f} /min",
            (x + 10, overlay_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, rr_color, 2
        )

        # Confidence
        conf_color = (0, int(255 * vs['confidence']), int(255 * (1 - vs['confidence'])))
        cv2.putText(
            frame, f"Confidence: {vs['confidence']:.1%}",
            (x + 10, overlay_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2
        )

    def _update_fps(self):
        """Update FPS counter"""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed

    def release(self):
        """Release resources"""
        if hasattr(self, 'pose_estimator'):
            self.pose_estimator.close()
        print("Layer 0 Computer Vision released")


def demo_layer0():
    """Demo Layer 0 with webcam"""
    print("=== Layer 0: Computer Vision Demo ===")
    print("Press 'q' to quit")

    # Initialize Layer 0
    layer0 = Layer0ComputerVision(use_gpu=True)

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        annotated_frame, persons = layer0.process_frame(frame)

        # Display
        cv2.imshow('WiFi-Vision-Pro: Layer 0 - Computer Vision', annotated_frame)

        # Print vital signs to console
        for person in persons:
            if person.vital_signs:
                print(f"Person {person.id}: "
                      f"HR={person.vital_signs['heart_rate']:.1f} BPM, "
                      f"RR={person.vital_signs['respiratory_rate']:.1f} /min")

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    layer0.release()


if __name__ == "__main__":
    demo_layer0()
