# ESP32-S3 CSI Validation Guide

## 🎯 Objective

Validate Channel State Information (CSI) extraction from real ESP32-S3 hardware to replace simulated data in `layer1_wifi_csi.py`.

**Priority:** 🔴 **P0 BLOCKER** - Project cannot proceed to commercial phase without this.

---

## 📋 Prerequisites

### Hardware ($30-50 total)
- **3x ESP32-S3-DevKitC-1** ($10 each)
  - [Amazon](https://www.amazon.com/s?k=ESP32-S3-DevKitC)
  - [AliExpress](https://www.aliexpress.com/w/wholesale-ESP32-S3-DevKitC.html)
  - [Mouser](https://www.mouser.com/c/semiconductors/embedded-processors-controllers/embedded-system-on-chip-soc/?q=ESP32-S3)
- **3x USB-C cables** (for programming/power)
- **Optional:** 3x external WiFi antennas (IPEX connector)

### Software
- **ESP-IDF v5.0+** (Espressif IoT Development Framework)
- **Python 3.8+** (for data processing)
- **Git** (for cloning repositories)

### Operating System
- Linux (Ubuntu 20.04+ recommended)
- macOS (10.15+)
- Windows 10/11 (with WSL2 recommended)

---

## 🔧 Installation Steps

### 1. Install ESP-IDF

#### Linux/macOS:
```bash
# Install dependencies
sudo apt-get install git wget flex bison gperf python3 python3-pip \
    python3-venv cmake ninja-build ccache libffi-dev libssl-dev \
    dfu-util libusb-1.0-0

# Clone ESP-IDF
mkdir -p ~/esp
cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
git checkout release/v5.1  # Use stable release

# Install ESP-IDF tools
./install.sh esp32s3

# Setup environment
. ./export.sh
```

#### Windows (WSL2):
```bash
# Install WSL2 Ubuntu first, then follow Linux steps
wsl --install -d Ubuntu-22.04
```

### 2. Clone ESP-CSI Repository

```bash
cd ~/esp
git clone https://github.com/espressif/esp-csi.git
cd esp-csi
```

**Important:** ESP-CSI is Espressif's official CSI extraction framework.

### 3. Configure ESP-CSI Project

```bash
cd examples/get-started/csi_recv_router

# Set build target
idf.py set-target esp32s3

# Configure project
idf.py menuconfig
```

**Configuration settings:**
- Navigate to `Component config → Wi-Fi`
- Enable `WiFi CSI`
- Set CSI data rate to `1000 Hz`
- Configure antenna settings (3 TX, 3 RX for MIMO)

### 4. Build and Flash Firmware

```bash
# Build firmware
idf.py build

# Connect ESP32-S3 via USB-C
# Find port (usually /dev/ttyUSB0 or /dev/ttyACM0)
ls /dev/tty*

# Flash firmware
idf.py -p /dev/ttyUSB0 flash

# Monitor output
idf.py -p /dev/ttyUSB0 monitor
```

**Expected output:**
```
CSI initialization successful
WiFi connected to <YOUR_SSID>
CSI callback registered
CSI data streaming...
```

---

## 📊 CSI Data Format

### Raw CSI Packet Structure

```c
typedef struct {
    wifi_pkt_rx_ctrl_t rx_ctrl;  // RX control info
    uint8_t mac[6];               // Source MAC address
    int8_t rssi;                  // RSSI (dBm)
    uint8_t rate;                 // PHY rate
    uint8_t sig_mode;             // Signal mode
    uint8_t mcs;                  // Modulation coding scheme
    uint8_t cwb;                  // Channel bandwidth
    uint8_t smoothing;            // Smoothing recommended
    uint8_t not_sounding;         // Not sounding PPDU
    uint8_t aggregation;          // AMPDU aggregation
    uint8_t stbc;                 // STBC
    uint8_t fec_coding;           // FEC coding
    uint8_t sgi;                  // Short GI
    int8_t noise_floor;           // Noise floor (dBm)
    uint16_t ampdu_cnt;           // AMPDU count
    uint16_t channel;             // Channel number
    uint16_t secondary_channel;   // Secondary channel
    uint32_t timestamp;           // Local timestamp (us)
    uint8_t ant;                  // Antenna number
    uint16_t sig_len;             // Signal length
    uint16_t rx_state;            // RX state

    // CSI data (complex numbers)
    int8_t buf[384];              // 128 subcarriers × 3 bytes (I,Q,pilot)
    uint16_t len;                 // CSI data length
    uint16_t first_word;          // First word invalid flag
} wifi_csi_info_t;
```

### CSI Subcarrier Data

**802.11n (HT20 - 20 MHz bandwidth):**
- **Subcarriers:** 52 (48 data + 4 pilots)
- **MIMO:** Up to 3×3 (9 CSI streams)
- **Format:** Complex I/Q pairs per subcarrier

**802.11ac (VHT80 - 80 MHz bandwidth):**
- **Subcarriers:** 242 (234 data + 8 pilots)
- **MIMO:** Up to 3×3
- **Higher spatial resolution**

### Data Extraction

```python
def parse_csi_data(raw_bytes):
    """Parse raw CSI bytes to complex numpy array"""
    # CSI format: [I0, Q0, pilot0, I1, Q1, pilot1, ...]
    num_subcarriers = len(raw_bytes) // 3

    csi_complex = np.zeros(num_subcarriers, dtype=np.complex128)

    for i in range(num_subcarriers):
        I = np.int8(raw_bytes[i*3])      # In-phase
        Q = np.int8(raw_bytes[i*3 + 1])  # Quadrature
        # pilot ignored for now

        csi_complex[i] = I + 1j * Q

    return csi_complex
```

---

## 🔌 Integration with layer1_wifi_csi.py

### Current Code (Simulated):

```python
# layer1_wifi_csi.py line 207-229
def _simulate_csi_loop(self):
    """Simulate CSI data for testing"""
    t = 0
    while self.is_receiving:
        csi = self._generate_simulated_csi(t)
        # ... store in buffer
```

### Modified Code (Real Hardware):

```python
def _receive_csi_loop(self):
    """Receive CSI packets from ESP32"""
    while self.is_receiving:
        try:
            # Read from ESP32 serial/UDP
            raw_packet = self.esp32_socket.recv(1024)

            # Parse packet
            timestamp = struct.unpack('<d', raw_packet[0:8])[0]
            rssi = struct.unpack('<f', raw_packet[8:12])[0]
            num_subcarriers = struct.unpack('<H', raw_packet[12:14])[0]

            # Extract CSI data
            csi_bytes = raw_packet[14:14+num_subcarriers*3]
            csi_complex = self._parse_csi_bytes(csi_bytes)

            # Create CSI packet
            packet = CSIPacket(
                timestamp=timestamp,
                subcarriers=csi_complex,
                rssi=rssi,
                noise_floor=-95.0,
                channel=36,
                bandwidth=20,
                tx_antenna=0,
                rx_antenna=0
            )

            self.csi_buffer.append(packet)

        except Exception as e:
            print(f"CSI receive error: {e}")
            continue
```

---

## 🧪 Validation Tests

### Test 1: Basic CSI Extraction

**Setup:**
- 1 ESP32-S3 as transmitter
- 1 ESP32-S3 as receiver
- Distance: 3 meters, line-of-sight

**Procedure:**
1. Power on both devices
2. Wait for WiFi connection
3. Observe CSI data streaming
4. Record 60 seconds of data

**Success criteria:**
- ✅ CSI packets received at 1000 Hz
- ✅ RSSI in expected range (-30 to -70 dBm)
- ✅ Subcarrier data non-zero
- ✅ No packet loss > 5%

### Test 2: Through-Wall Detection

**Setup:**
- 2 ESP32-S3 (TX/RX) on opposite sides of drywall
- Wall thickness: 10-15 cm
- No person in room (baseline)

**Procedure:**
1. Capture 30s CSI baseline (empty room)
2. Person enters room, stands still (30s)
3. Person walks slowly (30s)
4. Person leaves room (30s)

**Success criteria:**
- ✅ CSI amplitude/phase changes when person enters
- ✅ Motion detected during walking
- ✅ Returns to baseline when person leaves
- ✅ SNR sufficient for breathing detection (>10 dB)

### Test 3: Vital Signs Extraction

**Setup:**
- Person sitting 2-3 meters from ESP32 array
- Ground truth: Pulse oximeter + chest belt

**Procedure:**
1. Record 5 minutes of CSI data
2. Simultaneously record HR (pulse ox) and RR (chest belt)
3. Process CSI to extract vital signs
4. Compare with ground truth

**Success criteria:**
- ✅ Heart rate error < 5 BPM (vs pulse ox)
- ✅ Respiratory rate error < 2 breaths/min
- ✅ Correlation coefficient > 0.8
- ✅ 90%+ time samples within acceptable range

---

## 📈 Performance Benchmarks

### Expected CSI Characteristics

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| **Sampling rate** | 1000 Hz | 500 Hz | <300 Hz |
| **RSSI (LOS)** | -40 dBm | -60 dBm | <-70 dBm |
| **RSSI (through wall)** | -60 dBm | -75 dBm | <-80 dBm |
| **SNR** | >20 dB | >10 dB | <5 dB |
| **Breathing detection** | 95% | 85% | <70% |
| **Heart rate detection** | 85% | 70% | <50% |

### Penetration Depth (from literature)

| Material | Typical Loss | Max Penetration |
|----------|--------------|-----------------|
| **Drywall** | 3-5 dB | 30-50 cm ✅ |
| **Wood** | 5-10 dB | 20-30 cm |
| **Brick** | 10-15 dB | 15-20 cm |
| **Concrete** | 15-25 dB | 5-10 cm ⚠️ |
| **Metal** | 40-60 dB | 0 cm ❌ |

---

## 🐛 Troubleshooting

### Problem: No CSI data received

**Possible causes:**
- ESP32 not in monitor mode
- Firmware not flashed correctly
- WiFi channel mismatch

**Solutions:**
```bash
# Re-flash with erase
idf.py -p /dev/ttyUSB0 erase-flash
idf.py -p /dev/ttyUSB0 flash

# Check WiFi config
idf.py menuconfig
# → Component config → Wi-Fi → CSI enable = Y

# Monitor logs
idf.py monitor
```

### Problem: CSI data all zeros

**Possible causes:**
- No WiFi traffic
- CSI callback not registered
- Wrong antenna configuration

**Solutions:**
- Verify WiFi connection active
- Check CSI enable flag in firmware
- Test with ping traffic: `ping -i 0.01 <ESP32_IP>`

### Problem: Low SNR

**Possible causes:**
- Distance too far
- Multiple walls
- 2.4 GHz interference

**Solutions:**
- Reduce distance (start at 2-3m)
- Use 5 GHz band (less interference)
- Add external antennas
- Increase TX power in menuconfig

---

## 📦 Deliverables

After successful validation:

1. **Working ESP32 setup**
   - 3 devices configured and tested
   - Firmware flashed and verified
   - Network communication stable

2. **10 minutes real CSI data**
   - Saved as `.npy` files
   - Includes person movement scenarios
   - Ground truth annotations

3. **Integration code**
   - Modified `layer1_wifi_csi.py`
   - ESP32 communication module
   - Data parsing functions

4. **Validation report**
   - SNR measurements
   - Through-wall penetration results
   - Vital signs accuracy vs ground truth

---

## 🚀 Next Steps After Validation

1. **If successful (SNR >10 dB, penetration >20cm):**
   - ✅ Proceed to dataset collection (Issue #4)
   - ✅ Train fusion model with real data
   - ✅ Demo video with real hardware (Issue #2)
   - ✅ Fundraising with validated tech

2. **If partial success (SNR 5-10 dB):**
   - 🟡 Optimize antenna placement
   - 🟡 Consider 3.5 GHz band (better penetration)
   - 🟡 Add signal amplification
   - 🟡 Limit to single-wall scenarios

3. **If failure (SNR <5 dB or penetration <10cm):**
   - 🔴 **PIVOT:** Camera-only medical device
   - 🔴 Focus on Layer 0 + Layer 3 (no WiFi)
   - 🔴 Still viable: remote monitoring via rPPG
   - 🔴 Lower capital requirement, faster to market

---

## 📚 References

- [ESP-IDF Programming Guide](https://docs.espressif.com/projects/esp-idf/en/latest/)
- [ESP-CSI GitHub](https://github.com/espressif/esp-csi)
- [CSI Toolkit (MATLAB)](https://github.com/lubingxian/CSI-Toolkit)
- [WiFi Sensing Papers](https://github.com/awesome-wifi-sensing/awesome-wifi-sensing)

---

**Timeline:** 1-2 weeks
**Budget:** $50 (hardware)
**Risk:** HIGH - This is the critical path blocker

**Success = Green light for unicorn path 🦄**
**Failure = Pivot to camera-only (still valuable) 📹**
