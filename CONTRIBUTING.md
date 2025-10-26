# Contributing to WiFi-Vision-Pro

First off, **thank you** for considering contributing to WiFi-Vision-Pro! 🎉

This project aims to democratize medical imaging for 2.5 billion underserved people globally. Your contribution can literally **save lives**. 🏥

---

## 🌟 How You Can Help

### Critical Needs (High Priority)

1. **🔧 Hardware Engineers**
   - ESP32-S3 CSI extraction expertise
   - RF/antenna design
   - PCB layout for production units

2. **🧠 ML/AI Specialists**
   - WiFi CSI signal processing
   - Sensor fusion (camera + WiFi)
   - Medical AI model optimization

3. **🏥 Medical Device Experts**
   - FDA regulatory pathway (510(k))
   - Clinical study design
   - ISO 13485 quality systems

4. **📊 Data Scientists**
   - Dataset collection protocols
   - Statistical validation
   - Clinical accuracy benchmarking

5. **🎨 UX/UI Designers**
   - Medical dashboard design
   - Clinician workflow optimization
   - Patient-facing interfaces

### All Contributions Welcome

- 🐛 Bug reports
- 📝 Documentation improvements
- 🧪 Test coverage
- 💡 Feature suggestions
- 🎓 Educational content
- 🌍 Translations (for global reach)

---

## 🚀 Getting Started

### 1. Set Up Development Environment

```bash
# Clone repository
git clone https://github.com/Yatrogenesis/WiFi-Vision-Pro.git
cd WiFi-Vision-Pro

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_pro.txt

# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests to verify setup
pytest tests/
```

### 2. Choose an Issue

Browse [open issues](https://github.com/Yatrogenesis/WiFi-Vision-Pro/issues) and look for:
- `good first issue` - Great for newcomers
- `help wanted` - We need community help
- `priority: critical` - Urgent issues

Comment on the issue to claim it:
> "I'd like to work on this. ETA: X days."

---

## 📋 Development Workflow

### Branching Strategy

```
main (production-ready code)
  ↓
develop (integration branch)
  ↓
feature/your-feature-name (your work)
```

### Step-by-Step Process

1. **Create a branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/my-awesome-feature
   ```

2. **Make your changes**
   - Write code following style guide (see below)
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   ```bash
   # Run all tests
   pytest tests/

   # Check code coverage
   pytest --cov=src --cov-report=term

   # Run linters
   black src/
   flake8 src/
   ```

4. **Commit with clear message**
   ```bash
   git add .
   git commit -m "feat: Add through-wall penetration test

   - Implement drywall penetration benchmark
   - Add SNR measurement utilities
   - Update ESP32 validation guide

   Closes #42"
   ```

   **Commit message format:**
   ```
   type(scope): Subject line (max 50 chars)

   Body explaining what and why (wrap at 72 chars).

   Closes #issue_number
   ```

   **Types:**
   - `feat`: New feature
   - `fix`: Bug fix
   - `docs`: Documentation changes
   - `test`: Adding tests
   - `refactor`: Code restructuring
   - `perf`: Performance improvements
   - `style`: Code style changes (formatting)
   - `chore`: Maintenance tasks

5. **Push and create Pull Request**
   ```bash
   git push origin feature/my-awesome-feature
   ```

   Then go to GitHub and create a PR to `develop` branch.

---

## 🎨 Code Style Guide

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Good ✅
def process_csi_data(
    csi_buffer: np.ndarray,
    sample_rate: float = 1000.0,
    filter_type: str = "bandpass"
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Process CSI data for vital signs extraction.

    Args:
        csi_buffer: Raw CSI data (T×Tx×Rx×Subcarriers)
        sample_rate: Sampling rate in Hz
        filter_type: Type of filter to apply

    Returns:
        Tuple of (processed_csi, metrics_dict)

    Raises:
        ValueError: If csi_buffer is empty
    """
    if len(csi_buffer) == 0:
        raise ValueError("CSI buffer cannot be empty")

    # Your implementation
    processed = apply_filter(csi_buffer, filter_type)

    return processed, {"snr": calculate_snr(processed)}


# Bad ❌
def process(data):  # Unclear name, no types, no docstring
    if not data: return None
    x=filter(data,"bp")  # No spaces, unclear variable names
    return x
```

### Key Rules

1. **Type hints everywhere**
   ```python
   def my_function(param: int) -> str:
       return str(param)
   ```

2. **Docstrings for all public functions**
   - Use Google style docstrings
   - Include Args, Returns, Raises

3. **Descriptive variable names**
   - `csi_amplitude` ✅ not `ca` ❌
   - `heart_rate_bpm` ✅ not `hr` ❌

4. **Max line length: 100 characters** (not 79)

5. **Use f-strings for formatting**
   ```python
   # Good ✅
   message = f"Heart rate: {hr:.1f} BPM"

   # Bad ❌
   message = "Heart rate: " + str(hr) + " BPM"
   ```

### Auto-formatting

```bash
# Format all code
black src/ tests/

# Check without modifying
black --check src/

# Lint
flake8 src/
```

---

## 🧪 Testing Requirements

### All PRs Must Include Tests

```python
# Example test structure
def test_my_new_feature():
    """Test that new feature works as expected"""
    # Arrange
    input_data = create_test_data()

    # Act
    result = my_new_feature(input_data)

    # Assert
    assert result.success == True
    assert result.accuracy > 0.85
```

### Test Coverage

- **Minimum:** 70% coverage for new code
- **Target:** 80%+ coverage
- **Critical paths:** 100% coverage (medical accuracy, fall detection)

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_layer0_computer_vision.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests (exclude hardware)
pytest -m "not hardware"
```

---

## 📝 Documentation Standards

### Code Documentation

1. **Module-level docstring**
   ```python
   """
   layer0_computer_vision.py

   Computer vision module for person detection and vital signs estimation.

   This module provides:
   - YOLOv8 person detection
   - MediaPipe pose estimation
   - Remote photoplethysmography (rPPG)
   """
   ```

2. **Class docstrings**
   ```python
   class Layer0ComputerVision:
       """
       Layer 0: Standard Computer Vision

       Provides real-time person detection, tracking, and vital signs
       estimation using camera input.

       Attributes:
           detector: YOLOv8 model instance
           pose_estimator: MediaPipe Pose instance
           tracked_persons: Dict of currently tracked persons

       Example:
           >>> layer0 = Layer0ComputerVision()
           >>> frame = cv2.imread('person.jpg')
           >>> annotated, persons = layer0.process_frame(frame)
           >>> print(f"Detected {len(persons)} persons")
       """
   ```

3. **Function docstrings**
   - See example above in "Code Style Guide"

### Markdown Documentation

- Use **clear headings** (H1-H3)
- Include **code examples**
- Add **diagrams/screenshots** where helpful
- Link to related docs
- Keep **line length ~80 chars** for readability

---

## 🔬 Hardware Testing

### ESP32-S3 Development

If you have ESP32-S3 hardware:

1. Follow [ESP32_VALIDATION_GUIDE.md](ESP32_VALIDATION_GUIDE.md)
2. Mark hardware tests with `@pytest.mark.hardware`
   ```python
   @pytest.mark.hardware
   def test_real_esp32_csi_extraction():
       """This test requires physical ESP32"""
       # Your hardware test
   ```
3. Share your results in the [Hardware Discussion](https://github.com/Yatrogenesis/WiFi-Vision-Pro/discussions/categories/hardware)

### Contributing Hardware Data

If you collect CSI data with real hardware:

1. **Dataset format:**
   ```
   your_dataset/
   ├── metadata.json (description, setup, timestamps)
   ├── csi_raw/
   │   └── *.npy (raw CSI data)
   ├── ground_truth/
   │   ├── vital_signs.csv
   │   └── positions.json
   └── README.md
   ```

2. **Submit via:**
   - Small datasets (<100 MB): GitHub PR
   - Large datasets (>100 MB): Upload to Google Drive/Dropbox, share link in issue

3. **Include:**
   - Hardware setup description
   - Room dimensions and materials
   - Ground truth measurement method
   - Any anomalies or special conditions

---

## 🏥 Medical Device Compliance

### IMPORTANT: Medical Claims

⚠️ **This software is currently NOT FDA-cleared and NOT for clinical use.**

When contributing:

- ❌ **DO NOT** make medical claims in code/docs
- ❌ **DO NOT** use "diagnose", "treat", "cure"
- ✅ **DO** use "estimate", "monitor", "assess"
- ✅ **DO** include disclaimers in medical-facing code

**Example:**
```python
# Good ✅
"""
Estimate vital signs for wellness monitoring.

DISCLAIMER: Not for medical use. Consult healthcare provider.
"""

# Bad ❌
"""
Diagnose heart conditions using AI.  # Medical claim!
"""
```

### Privacy & Security

- ❌ **NEVER** commit patient data, PHI, or PII
- ❌ **NEVER** hardcode API keys, credentials
- ✅ **USE** `.gitignore` for sensitive files
- ✅ **USE** environment variables for secrets

---

## 🤝 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment.

**Expected behavior:**
- ✅ Be respectful and constructive
- ✅ Welcome newcomers warmly
- ✅ Give credit where it's due
- ✅ Focus on what's best for the community

**Unacceptable behavior:**
- ❌ Harassment, discrimination, or trolling
- ❌ Personal attacks or insults
- ❌ Spam or self-promotion
- ❌ Sharing others' private information

**Enforcement:** Violations may result in temporary or permanent ban.

**Report issues:** email conduct@yatrogenesis.com

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** General questions, ideas
- **Discord** (coming soon): Real-time chat
- **Twitter:** [@WiFiVisionPro](https://twitter.com/WiFiVisionPro) (updates)

---

## 🎓 Learning Resources

### New to WiFi Sensing?

- [Awesome WiFi Sensing](https://github.com/awesome-wifi-sensing/awesome-wifi-sensing) - Papers & tutorials
- [ESP-CSI Documentation](https://github.com/espressif/esp-csi)
- [MUSIC Algorithm Tutorial](https://www.gaussianwaves.com/2015/04/music-algorithm/)

### New to Medical Devices?

- [FDA 510(k) Guide](https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-notification-510k)
- [ISO 13485 Overview](https://www.iso.org/iso-13485-medical-devices.html)
- [IEC 62304 (Medical Software)](https://www.iec.ch/medical-electrical-equipment-software)

### New to Contributing?

- [First Contributions Guide](https://github.com/firstcontributions/first-contributions)
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

---

## 🏆 Recognition

### Contributors

All contributors are recognized in:
- `CONTRIBUTORS.md` (alphabetical list)
- GitHub contributors page
- Release notes (if applicable)

### Significant Contributions

Outstanding contributions may be recognized with:
- **Core Contributor** badge
- **Advisory Board** invitation (for sustained contributions)
- **Co-authorship** on academic papers
- **Equity grants** (if/when company fundraises)

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

**Important:** For medical device commercialization, we may need Contributor License Agreements (CLAs) in the future. We'll notify all contributors if this becomes necessary.

---

## ❓ Questions?

**Before opening an issue, check:**
1. [FAQ](https://github.com/Yatrogenesis/WiFi-Vision-Pro/wiki/FAQ) (coming soon)
2. [Existing issues](https://github.com/Yatrogenesis/WiFi-Vision-Pro/issues)
3. [Discussions](https://github.com/Yatrogenesis/WiFi-Vision-Pro/discussions)

**Still stuck?**
- Open a [GitHub Discussion](https://github.com/Yatrogenesis/WiFi-Vision-Pro/discussions)
- Tag maintainers: @Yatrogenesis

---

## 🎯 Contribution Ideas

Not sure where to start? Try these:

### Beginner-Friendly
- [ ] Add type hints to legacy code
- [ ] Write unit tests for uncovered functions
- [ ] Improve docstrings
- [ ] Fix typos in documentation
- [ ] Add examples to README

### Intermediate
- [ ] Implement missing Layer 1 tests
- [ ] Create demo Jupyter notebooks
- [ ] Build Docker container for easy setup
- [ ] Add support for additional cameras (RTSP, IP cameras)
- [ ] Create CLI tool for batch processing

### Advanced
- [ ] ESP32-S3 firmware development
- [ ] Train sensor fusion model with real data
- [ ] Optimize MUSIC algorithm for real-time performance
- [ ] Implement additional vital signs (blood pressure estimation)
- [ ] Create FDA-compliant design controls documentation

---

## 🚀 Thank You!

Your contributions make WiFi-Vision-Pro possible. Together, we're building the future of accessible healthcare for billions of people.

**Let's save lives through technology.** 🏥💙

---

**Questions about contributing?**
Open an issue or start a discussion. We're here to help! 😊

**Happy coding!** 👨‍💻👩‍💻
