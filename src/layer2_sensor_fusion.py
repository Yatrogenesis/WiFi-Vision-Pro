#!/usr/bin/env python3
"""
WiFi-Vision-Pro: Layer 2 - Sensor Fusion
Combines Camera (Layer 0) + WiFi CSI (Layer 1) for complete scene understanding
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass

from layer0_computer_vision import Person, Layer0ComputerVision
from layer1_wifi_csi import ThroughWallPerson, Layer1WiFiCSI


@dataclass
class FusedPerson:
    """Person with fused camera + WiFi data"""
    id: int
    visible: bool  # Can camera see them?
    position_2d: Tuple[int, int]  # Camera coordinates
    position_3d: Tuple[float, float, float]  # WiFi 3D position
    heart_rate: float  # BPM (best estimate)
    respiratory_rate: float  # /min (best estimate)
    confidence_camera: float
    confidence_wifi: float
    confidence_fused: float


class MultiModalFusionNetwork(nn.Module):
    """Neural network for fusing camera + WiFi features"""

    def __init__(self, camera_features=512, wifi_features=512):
        super().__init__()

        # Attention mechanism for adaptive weighting
        self.attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True
        )

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(camera_features + wifi_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # HR, RR, confidence_cam, confidence_wifi
        )

    def forward(self, camera_feat, wifi_feat):
        # Cross-modal attention
        attended, _ = self.attention(camera_feat, wifi_feat, wifi_feat)

        # Concatenate and fuse
        fused = torch.cat([attended, wifi_feat], dim=-1)
        output = self.fusion(fused)

        return output


class Layer2SensorFusion:
    """
    Layer 2: Sensor Fusion

    Intelligently combines:
    - Camera RGB (surface detection, high resolution)
    - WiFi CSI (through-wall, motion sensitive)

    For robust multi-modal tracking and vital signs
    """

    def __init__(self, use_neural_fusion=True):
        # Initialize sub-layers
        self.layer0 = Layer0ComputerVision(use_gpu=True)
        self.layer1 = Layer1WiFiCSI(room_dimensions=(5.0, 5.0, 3.0))

        # Fusion network
        self.use_neural_fusion = use_neural_fusion
        if use_neural_fusion:
            self.fusion_net = MultiModalFusionNetwork()
            self.fusion_net.eval()  # Inference mode

        # Person association
        self.fused_persons: Dict[int, FusedPerson] = {}
        self.next_id = 0

        print("✓ Sensor Fusion Layer initialized")
        print(f"  Using neural fusion: {use_neural_fusion}")

    def start(self):
        """Start both layers"""
        self.layer1.start_esp32_stream()
        print("✓ All layers started")

    def process(self, camera_frame: np.ndarray) -> Tuple[np.ndarray, List[FusedPerson]]:
        """
        Process camera frame + WiFi CSI for fused result

        Args:
            camera_frame: BGR image from camera

        Returns:
            annotated_frame: Visualization
            fused_persons: List of persons with fused data
        """
        # Step 1: Process camera
        _, camera_persons = self.layer0.process_frame(camera_frame)

        # Step 2: Process WiFi CSI
        spatial_map, wifi_persons = self.layer1.process_csi()

        # Step 3: Associate camera and WiFi detections
        fused_persons = self._associate_detections(camera_persons, wifi_persons)

        # Step 4: Fuse vital signs estimates
        for person in fused_persons:
            self._fuse_vital_signs(person, camera_persons, wifi_persons)

        # Step 5: Create visualization
        annotated_frame = self._create_fusion_visualization(
            camera_frame, spatial_map, fused_persons
        )

        return annotated_frame, fused_persons

    def _associate_detections(
        self,
        camera_persons: List[Person],
        wifi_persons: List[ThroughWallPerson]
    ) -> List[FusedPerson]:
        """Associate camera and WiFi detections to same persons"""
        fused = []

        # Simple association: match based on 2D projection of 3D WiFi position
        for cam_person in camera_persons:
            # Get camera center
            x_cam, y_cam = cam_person.center

            # Find closest WiFi detection (if any)
            best_wifi = None
            best_distance = float('inf')

            for wifi_person in wifi_persons:
                # Project 3D WiFi position to 2D camera plane (simplified)
                x_wifi_2d = wifi_person.position[0] * 100  # meters to pixels (rough)
                y_wifi_2d = wifi_person.position[1] * 100

                dist = np.sqrt((x_cam - x_wifi_2d)**2 + (y_cam - y_wifi_2d)**2)

                if dist < best_distance and dist < 200:  # 200 pixel threshold
                    best_distance = dist
                    best_wifi = wifi_person

            # Create fused person
            person = FusedPerson(
                id=cam_person.id,
                visible=True,
                position_2d=cam_person.center,
                position_3d=best_wifi.position if best_wifi else (0, 0, 0),
                heart_rate=0,
                respiratory_rate=0,
                confidence_camera=cam_person.confidence,
                confidence_wifi=best_wifi.confidence if best_wifi else 0,
                confidence_fused=0
            )
            fused.append(person)

        # Add WiFi-only detections (people behind walls)
        for wifi_person in wifi_persons:
            # Check if already matched
            if any(f.position_3d == wifi_person.position for f in fused):
                continue

            person = FusedPerson(
                id=self.next_id,
                visible=False,  # Not visible to camera (behind wall!)
                position_2d=(0, 0),
                position_3d=wifi_person.position,
                heart_rate=0,
                respiratory_rate=0,
                confidence_camera=0,
                confidence_wifi=wifi_person.confidence,
                confidence_fused=0
            )
            self.next_id += 1
            fused.append(person)

        return fused

    def _fuse_vital_signs(
        self,
        fused_person: FusedPerson,
        camera_persons: List[Person],
        wifi_persons: List[ThroughWallPerson]
    ):
        """Fuse vital signs from camera and WiFi"""
        cam_hr, cam_rr = 0, 0
        wifi_hr, wifi_rr = 0, 0

        # Get camera vital signs
        cam_match = next((p for p in camera_persons if p.center == fused_person.position_2d), None)
        if cam_match and cam_match.vital_signs:
            cam_hr = cam_match.vital_signs['heart_rate']
            cam_rr = cam_match.vital_signs['respiratory_rate']

        # Get WiFi vital signs
        wifi_match = next((p for p in wifi_persons if p.position == fused_person.position_3d), None)
        if wifi_match:
            wifi_hr = wifi_match.heart_rate
            wifi_rr = wifi_match.breathing_rate

        # Weighted fusion based on confidence
        w_cam = fused_person.confidence_camera
        w_wifi = fused_person.confidence_wifi
        total_weight = w_cam + w_wifi

        if total_weight > 0:
            fused_person.heart_rate = (cam_hr * w_cam + wifi_hr * w_wifi) / total_weight
            fused_person.respiratory_rate = (cam_rr * w_cam + wifi_rr * w_wifi) / total_weight
            fused_person.confidence_fused = (w_cam + w_wifi) / 2

    def _create_fusion_visualization(
        self,
        camera_frame: np.ndarray,
        spatial_map: np.ndarray,
        fused_persons: List[FusedPerson]
    ) -> np.ndarray:
        """Create split-screen visualization"""
        h, w = camera_frame.shape[:2]

        # Left: Camera view
        left = camera_frame.copy()

        # Right: WiFi heatmap
        heatmap = self.layer1.visualize_spatial_map(spatial_map)
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Combine side-by-side
        combined = np.hstack([left, heatmap_resized])

        # Annotate fused persons
        for person in fused_persons:
            if person.visible:
                # Draw on camera side
                x, y = person.position_2d
                cv2.circle(combined, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(
                    combined, f"Fused ID{person.id}",
                    (x, y-20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2
                )
            else:
                # WiFi-only detection (behind wall)
                x_3d, y_3d, z_3d = person.position_3d
                x_2d = int((x_3d / 5.0) * w) + w  # Project to right side
                y_2d = int((y_3d / 5.0) * h)

                cv2.circle(combined, (x_2d, y_2d), 15, (0, 0, 255), 3)
                cv2.putText(
                    combined, f"THROUGH WALL ID{person.id}",
                    (x_2d-80, y_2d-20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2
                )

        return combined

    def release(self):
        """Release resources"""
        self.layer0.release()
        self.layer1.stop()


def demo_layer2():
    """Demo Layer 2 sensor fusion"""
    print("=== Layer 2: Sensor Fusion Demo ===")
    print("Press 'q' to quit")

    fusion = Layer2SensorFusion(use_neural_fusion=False)
    fusion.start()

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process fusion
        annotated, fused_persons = fusion.process(frame)

        # Display
        cv2.imshow('WiFi-Vision-Pro: Layer 2 - Sensor Fusion', annotated)

        # Print fused vital signs
        for person in fused_persons:
            visibility = "VISIBLE" if person.visible else "THROUGH WALL"
            print(f"Person {person.id} ({visibility}): "
                  f"HR={person.heart_rate:.1f}, RR={person.respiratory_rate:.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    fusion.release()


if __name__ == "__main__":
    demo_layer2()
