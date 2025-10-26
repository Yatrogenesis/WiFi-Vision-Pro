# WiFi-Vision-Pro Testing Plan

## 🎯 Objective

Establish comprehensive testing framework to ensure medical-grade quality and prevent regressions.

**Target:** 80%+ code coverage, <1% defect rate

---

## 📋 Testing Pyramid

```
        /\
       /  \         E2E Tests (5%)
      /----\        - Full system validation
     /      \       - Hardware-in-the-loop
    /--------\      Integration Tests (15%)
   /          \     - Multi-component workflows
  /------------\    - API contracts
 /______________\   Unit Tests (80%)
                    - Individual functions
                    - Edge cases
```

---

## 🧪 Unit Tests

### Layer 0: Computer Vision

**File:** `tests/test_layer0_computer_vision.py`

```python
import pytest
import numpy as np
from src.layer0_computer_vision import Layer0ComputerVision, Person

class TestPersonDetection:
    def test_yolo_detection_valid_image(self):
        """Test YOLOv8 person detection on standard image"""
        layer0 = Layer0ComputerVision()

        # Load test image with 2 people
        image = cv2.imread('tests/fixtures/two_people.jpg')

        annotated, persons = layer0.process_frame(image)

        assert len(persons) == 2
        assert all(p.confidence > 0.7 for p in persons)
        assert all(p.bbox[2] > 0 and p.bbox[3] > 0 for p in persons)

    def test_person_tracking_across_frames(self):
        """Test person ID persistence across frames"""
        layer0 = Layer0ComputerVision()

        # Frame 1: Person at (100, 100)
        frame1 = create_test_image_with_person(100, 100)
        _, persons1 = layer0.process_frame(frame1)
        id1 = persons1[0].id

        # Frame 2: Person moved to (110, 105) - should keep same ID
        frame2 = create_test_image_with_person(110, 105)
        _, persons2 = layer0.process_frame(frame2)

        assert persons2[0].id == id1  # Same person

    def test_vital_signs_estimation_accuracy(self):
        """Test rPPG vital signs vs ground truth"""
        layer0 = Layer0ComputerVision()

        # Load 60s video with known HR (72 BPM) and RR (16 /min)
        video_frames = load_test_video('tests/fixtures/vital_signs_test.mp4')

        for frame in video_frames:
            _, persons = layer0.process_frame(frame)

        estimated_hr = persons[0].vital_signs['heart_rate']
        estimated_rr = persons[0].vital_signs['respiratory_rate']

        # Medical-grade accuracy thresholds
        assert abs(estimated_hr - 72.0) < 5.0  # <5 BPM error
        assert abs(estimated_rr - 16.0) < 2.0  # <2 /min error

class TestPoseEstimation:
    def test_mediapipe_keypoints_detection(self):
        """Test MediaPipe pose landmark detection"""
        layer0 = Layer0ComputerVision()

        image = cv2.imread('tests/fixtures/standing_person.jpg')
        _, persons = layer0.process_frame(image)

        assert persons[0].keypoints is not None
        assert len(persons[0].keypoints) == 33  # MediaPipe landmarks

        # Check visibility of key joints
        nose, left_shoulder, right_shoulder = persons[0].keypoints[0], persons[0].keypoints[11], persons[0].keypoints[12]
        assert all(kp[2] > 0.5 for kp in [nose, left_shoulder, right_shoulder])  # High visibility

class TestPerformance:
    def test_fps_realtime_requirement(self):
        """Test that processing achieves >15 FPS"""
        layer0 = Layer0ComputerVision()

        image = cv2.imread('tests/fixtures/test_frame.jpg')

        import time
        start = time.time()

        for _ in range(30):  # Process 30 frames
            layer0.process_frame(image)

        elapsed = time.time() - start
        fps = 30 / elapsed

        assert fps >= 15.0  # Minimum real-time requirement
```

---

### Layer 1: WiFi CSI Sensing

**File:** `tests/test_layer1_wifi_csi.py`

```python
import pytest
import numpy as np
from src.layer1_wifi_csi import Layer1WiFiCSI, CSIPacket

class TestCSIProcessing:
    def test_csi_preprocessing_removes_static(self):
        """Test DC offset removal in CSI preprocessing"""
        layer1 = Layer1WiFiCSI()

        # Create CSI with known DC offset
        csi_with_dc = np.ones((1000, 3, 3, 52)) * (10 + 5j)  # Static component
        csi_with_dc += np.random.randn(1000, 3, 3, 52) * 0.1  # Small variation

        csi_clean = layer1._preprocess_csi(csi_with_dc)

        # Mean should be ~0 after DC removal
        assert np.abs(np.mean(csi_clean)) < 0.01

    def test_spatial_mapping_person_detection(self):
        """Test MUSIC algorithm detects person in room"""
        layer1 = Layer1WiFiCSI()

        # Simulate CSI with person at (2.5m, 2.5m, 1.5m)
        for _ in range(1000):
            packet = layer1._generate_simulated_csi(t=0.001 * _)
            layer1.csi_buffer.append(CSIPacket(
                timestamp=time.time(),
                subcarriers=packet,
                rssi=-50.0,
                noise_floor=-95.0,
                channel=36,
                bandwidth=20,
                tx_antenna=0,
                rx_antenna=0
            ))

        spatial_map, persons = layer1.process_csi()

        assert len(persons) >= 1  # At least one person detected

        # Person should be near (2.5, 2.5, 1.5)
        detected_pos = persons[0].position
        assert abs(detected_pos[0] - 2.5) < 0.5  # Within 50cm
        assert abs(detected_pos[1] - 2.5) < 0.5
        assert abs(detected_pos[2] - 1.5) < 0.5

    def test_breathing_rate_extraction(self):
        """Test breathing rate estimation from CSI phase"""
        layer1 = Layer1WiFiCSI()

        # Simulate 60s with 18 breaths/min (0.3 Hz)
        for t in np.linspace(0, 60, 60000):  # 1000 Hz
            # CSI with breathing modulation
            breathing_signal = 0.01 * np.sin(2 * np.pi * 0.3 * t)
            # ... (full CSI generation)

        _, persons = layer1.process_csi()

        estimated_br = persons[0].breathing_rate

        assert abs(estimated_br - 18.0) < 2.0  # <2 breaths/min error

class TestThroughWallPenetration:
    @pytest.mark.hardware  # Requires real ESP32
    def test_drywall_penetration_real_hardware(self):
        """Test signal penetration through drywall"""
        # This test requires physical setup
        pass  # TODO: Implement when hardware available

class TestNetworking:
    def test_esp32_socket_connection(self):
        """Test socket connection to ESP32"""
        layer1 = Layer1WiFiCSI()

        # Mock ESP32 server
        mock_server = create_mock_esp32_server(port=8080)

        success = layer1.start_esp32_stream(esp32_ip="127.0.0.1", port=8080)

        assert success
        assert layer1.is_receiving

    def test_csi_packet_parsing(self):
        """Test parsing of raw ESP32 CSI packets"""
        layer1 = Layer1WiFiCSI()

        # Create mock packet
        raw_packet = create_mock_csi_packet(
            timestamp=time.time(),
            rssi=-45.0,
            num_subcarriers=52
        )

        parsed = layer1._parse_esp32_packet(raw_packet)

        assert parsed.subcarriers.shape == (52,)
        assert parsed.rssi == -45.0
```

---

### Layer 2: Sensor Fusion

**File:** `tests/test_layer2_sensor_fusion.py`

```python
import pytest
from src.layer2_sensor_fusion import Layer2SensorFusion, FusedPerson

class TestSensorFusion:
    def test_camera_wifi_association(self):
        """Test matching camera and WiFi detections"""
        fusion = Layer2SensorFusion(use_neural_fusion=False)

        # Camera detects person at (100, 100) pixels
        camera_persons = [create_mock_camera_person(100, 100)]

        # WiFi detects person at (1.0, 1.0, 1.5) meters
        wifi_persons = [create_mock_wifi_person(1.0, 1.0, 1.5)]

        fused = fusion._associate_detections(camera_persons, wifi_persons)

        assert len(fused) == 1
        assert fused[0].visible == True  # Both detected
        assert fused[0].confidence_camera > 0
        assert fused[0].confidence_wifi > 0

    def test_vital_signs_fusion_weighted_average(self):
        """Test weighted averaging of vital signs"""
        fusion = Layer2SensorFusion()

        # Camera: HR=70, confidence=0.8
        # WiFi: HR=74, confidence=0.9
        # Expected: (70*0.8 + 74*0.9) / (0.8+0.9) = 72.35

        fused_person = create_mock_fused_person(
            camera_hr=70, camera_conf=0.8,
            wifi_hr=74, wifi_conf=0.9
        )

        fusion._fuse_vital_signs(fused_person, camera_persons, wifi_persons)

        assert abs(fused_person.heart_rate - 72.35) < 0.1

    @pytest.mark.slow  # Neural network test
    def test_neural_fusion_inference(self):
        """Test neural fusion network forward pass"""
        fusion = Layer2SensorFusion(use_neural_fusion=True)

        camera_feat = torch.randn(1, 512)
        wifi_feat = torch.randn(1, 512)

        output = fusion.fusion_net(camera_feat, wifi_feat)

        assert output.shape == (1, 4)  # HR, RR, conf_cam, conf_wifi
```

---

### Layer 3: Medical Application

**File:** `tests/test_layer3_medical_application.py`

```python
import pytest
from src.layer3_medical_application import Layer3MedicalApplication, FallEvent

class TestClinicalVitalSigns:
    def test_bradycardia_alert_generation(self):
        """Test alert when heart rate < 50 BPM"""
        medical = Layer3MedicalApplication()

        # Create patient with HR=45 BPM (bradycardia)
        patient = create_mock_patient(heart_rate=45, respiratory_rate=16)

        vital_signs = medical._assess_vital_signs(patient)
        alerts = medical._generate_clinical_alerts(vital_signs, fall_event=None)

        assert any("BRADYCARDIA" in alert for alert in alerts)

    def test_fall_detection_sensitivity(self):
        """Test fall detection with rapid z-drop"""
        medical = Layer3MedicalApplication()

        # Simulate fall: z drops from 1.7m (standing) to 0.5m (ground) in 1s
        medical.last_position_z = 1.7

        for _ in range(30):  # 30 frames @ 30 FPS = 1 second
            patient = create_mock_patient(position_z=0.5)
            medical.process_frame(frame)  # Mock frame

        # After 5 seconds on ground, should detect fall
        time.sleep(5)
        clinical_data = medical.process_frame(frame)

        assert clinical_data['fall_event'] is not None
        assert clinical_data['fall_event'].severity in ['mild', 'moderate', 'severe']

    def test_hipaa_data_logging(self):
        """Test HIPAA-compliant data logging format"""
        medical = Layer3MedicalApplication(patient_id="TEST_001")

        vital_signs = create_mock_vital_signs(hr=72, rr=16)
        medical._log_clinical_data(vital_signs)

        # Read log file
        with open(medical.log_file, 'r') as f:
            log_entry = json.loads(f.readlines()[-1])

        # Verify required fields
        assert 'patient_id' in log_entry
        assert 'timestamp' in log_entry
        assert 'heart_rate' in log_entry
        assert log_entry['patient_id'] == "TEST_001"

class TestApneaDetection:
    def test_apnea_episode_detection(self):
        """Test detection of apnea (breathing cessation >10s)"""
        medical = Layer3MedicalApplication()
        medical.sleep_monitoring_enabled = True

        # Simulate apnea: RR drops to 0 for 15 seconds
        for _ in range(15):
            patient = create_mock_patient(respiratory_rate=0)
            medical.process_frame(frame)
            time.sleep(1)

        # Should detect apnea
        assert len(medical.apnea_episodes) > 0
        assert medical.apnea_episodes[-1].duration >= 10.0
```

---

## 🔗 Integration Tests

### End-to-End Workflow

**File:** `tests/integration/test_e2e_pipeline.py`

```python
class TestEndToEndPipeline:
    def test_full_pipeline_camera_to_clinical_report(self):
        """Test complete pipeline from camera input to clinical data"""
        # Initialize full stack
        medical_app = Layer3MedicalApplication()
        medical_app.start_monitoring()

        # Load 60-second test video
        video = cv2.VideoCapture('tests/fixtures/patient_resting_60s.mp4')

        results = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            clinical_data = medical_app.process_frame(frame)
            results.append(clinical_data)

        # Verify outputs
        assert len(results) > 0
        assert all('vital_signs' in r for r in results)

        # Check clinical report generation
        report = medical_app.generate_clinical_report(duration_hours=1)

        assert 'vital_signs_summary' in report
        assert report['vital_signs_summary']['heart_rate_mean'] > 0

    @pytest.mark.hardware
    def test_real_esp32_to_visualization(self):
        """Test with real ESP32 hardware"""
        # Requires physical ESP32 setup
        layer1 = Layer1WiFiCSI()
        success = layer1.start_esp32_stream(esp32_ip="192.168.1.100")

        if not success:
            pytest.skip("ESP32 hardware not available")

        # Wait for CSI buffer to fill
        time.sleep(5)

        spatial_map, persons = layer1.process_csi()

        assert spatial_map.shape[0] > 0  # Not empty
        # Person detection depends on actual environment
```

---

## ⚡ Performance Tests

### Benchmarks

**File:** `tests/performance/test_benchmarks.py`

```python
import pytest
from pytest_benchmark.plugin import benchmark

class TestPerformanceBenchmarks:
    def test_layer0_fps_benchmark(self, benchmark):
        """Benchmark Layer 0 processing FPS"""
        layer0 = Layer0ComputerVision()
        test_frame = cv2.imread('tests/fixtures/benchmark_frame.jpg')

        result = benchmark(layer0.process_frame, test_frame)

        # Should process in <67ms for 15 FPS
        assert benchmark.stats.mean < 0.067

    def test_layer1_csi_processing_latency(self, benchmark):
        """Benchmark CSI processing latency"""
        layer1 = Layer1WiFiCSI()

        # Fill buffer with 1000 samples
        for _ in range(1000):
            layer1.csi_buffer.append(create_mock_csi_packet())

        result = benchmark(layer1.process_csi)

        # Should complete in <100ms
        assert benchmark.stats.mean < 0.1

    def test_memory_usage_long_running(self):
        """Test memory doesn't leak over long runs"""
        import psutil
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        layer0 = Layer0ComputerVision()
        test_frame = cv2.imread('tests/fixtures/test_frame.jpg')

        # Run for 1000 frames
        for _ in range(1000):
            layer0.process_frame(test_frame)

            if _ % 100 == 0:
                gc.collect()  # Force garbage collection

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # Memory increase should be <100 MB for 1000 frames
        assert memory_increase < 100
```

---

## 🎯 Test Coverage

### Coverage Goals

| Component | Target | Current | Status |
|-----------|--------|---------|--------|
| Layer 0 | 85% | TBD | 🔴 Not tested |
| Layer 1 | 80% | TBD | 🔴 Not tested |
| Layer 2 | 75% | TBD | 🔴 Not tested |
| Layer 3 | 85% | TBD | 🔴 Not tested |
| **Overall** | **80%** | **TBD** | 🔴 **Critical** |

### Generate Coverage Report

```bash
# Install coverage tools
pip install pytest pytest-cov pytest-benchmark

# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

---

## 🚀 CI/CD Integration

### GitHub Actions Workflow

**File:** `.github/workflows/tests.yml`

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements_pro.txt
        pip install pytest pytest-cov pytest-benchmark

    - name: Run unit tests
      run: pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

---

## 📝 Test Fixtures

### Required Test Data

```
tests/
├── fixtures/
│   ├── images/
│   │   ├── one_person.jpg
│   │   ├── two_people.jpg
│   │   ├── standing_person.jpg
│   │   └── benchmark_frame.jpg
│   ├── videos/
│   │   ├── patient_resting_60s.mp4
│   │   ├── vital_signs_test.mp4 (with ground truth)
│   │   └── fall_detection_test.mp4
│   ├── csi_data/
│   │   ├── empty_room_baseline.npy
│   │   ├── person_standing.npy
│   │   ├── person_walking.npy
│   │   └── through_wall_test.npy
│   └── ground_truth/
│       ├── vital_signs_gt.csv
│       ├── positions_gt.json
│       └── fall_events_gt.json
```

---

## ✅ Acceptance Criteria

### Before Production Deployment

- [ ] **All unit tests passing** (100%)
- [ ] **Integration tests passing** (100%)
- [ ] **Code coverage ≥80%**
- [ ] **Performance benchmarks met**
  - Layer 0: ≥15 FPS
  - Layer 1: <100ms latency
  - Memory: <500 MB peak usage
- [ ] **No critical/high severity bugs**
- [ ] **CI/CD pipeline green**
- [ ] **Medical accuracy validated**
  - HR: <5 BPM error
  - RR: <2 /min error
  - Fall detection: >90% sensitivity

---

**Testing is critical for medical devices. Lives depend on quality.** 🏥

**Next step:** Implement tests starting with Layer 0 unit tests.

**Target date:** Week of Nov 4, 2025 (1 week sprint)
