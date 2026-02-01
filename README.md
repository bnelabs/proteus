# PROTEUS BLE Gateway

A **cross-platform** Python gateway for the **ST STEVAL-PROTEUS1** industrial vibration sensor. Connects via Bluetooth Low Energy (BLE), streams sensor data, calculates vibration health metrics, and detects anomalies using statistical baselines.

**Supports: Windows, macOS, Linux**

## Features

- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Auto-Discovery**: Scans and finds PROTEUS devices automatically
- **BLE Connection**: Auto-reconnect with exponential backoff
- **Sensor Data Streaming**: Accelerometer, gyroscope, FFT, velocity RMS, temperature, pressure
- **Vibration Analysis**:
  - ISO 20816 zone classification
  - Crest factor & kurtosis calculation
  - Bearing health index (0-100 score)
- **Anomaly Detection**: Learn baseline from healthy machine, detect deviations
- **Data Export**: MQTT to ThingsBoard, optional NATS support
- **Real-time Dashboard**: NiceGUI-based visualization (optional)

## Hardware

- **Sensor Board**: [STEVAL-PROTEUS1](https://www.st.com/en/evaluation-tools/steval-proteus1.html) (~$50 USD)
- **Host**: Any computer with Bluetooth (Windows 10+, macOS 10.15+, Linux with BlueZ)

## Quick Start

### 1. Install Dependencies

**macOS/Linux:**
```bash
cd proteus-gateway
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```cmd
cd proteus-gateway
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Find Your PROTEUS Device

```bash
# Scan for BLE devices
python proteus_gateway.py --scan
```

Output:
```
✓ PROTEUS DEVICES FOUND:
  Name:    PROTEUS
  Address: AA:BB:CC:DD:EE:FF   (Windows/Linux)
           2402245D-06F8-...   (macOS)
```

### 3. Run Gateway

**macOS** (usually auto-detects):
```bash
python proteus_gateway.py
```

**Windows/Linux** (use discovered address):
```bash
python proteus_gateway.py --address AA:BB:CC:DD:EE:FF
```

**With anomaly detection:**
```bash
python proteus_gateway.py --address YOUR_ADDRESS --mode learn
```

## Platform Notes

| Platform | Address Format | Example |
|----------|----------------|---------|
| **macOS** | UUID | `2402245D-06F8-8C06-6450-9C102D4D7CE6` |
| **Windows** | MAC | `AA:BB:CC:DD:EE:FF` |
| **Linux** | MAC | `AA:BB:CC:DD:EE:FF` |

- **macOS**: Usually works out of the box with auto-discovery
- **Windows**: Run `--scan` first to find your device's MAC address
- **Linux**: May need BlueZ installed (`sudo apt install bluez`)

## Anomaly Detection

The gateway includes a built-in anomaly detector that learns what "normal" looks like for your specific machine, then alerts when readings deviate significantly.

### How It Works

1. **Learn Mode**: Collect baseline data from a healthy machine
2. **Monitor Mode**: Compare real-time data against baseline, flag anomalies

### Step 1: Learn Baseline (Run on Healthy Machine)

```bash
python proteus_gateway.py --mode learn --baseline baseline.json
```

- Minimum: 500 samples (~8 minutes)
- Recommended: Run for 1-2 days for robust baseline
- Baseline auto-saves when minimum reached, and on Ctrl+C

**Output during learning:**
```
INFO - AnomalyDetector initialized in LEARN mode
INFO - Learning progress: 100 samples (20%)
INFO - Learning progress: 500 samples (100%)
INFO - Baseline calculated from 500 samples
INFO - Baseline saved to baseline.json
```

### Step 2: Monitor for Anomalies

```bash
python proteus_gateway.py --mode monitor --baseline baseline.json
```

**When anomaly detected:**
```
WARNING - ANOMALY DETECTED (score=0.75): velocity_rms=2.8500 (warning, z=3.2)
```

### Telemetry Output

**Learning mode:**
```json
{
  "velocity_rms": 0.85,
  "anomaly_mode": "learning",
  "anomaly_progress": 45.2,
  "anomaly_samples": 226
}
```

**Monitor mode:**
```json
{
  "velocity_rms": 2.85,
  "anomaly": true,
  "anomaly_score": 0.75,
  "anomaly_mode": "monitoring",
  "anomaly_reasons": ["velocity_rms"]
}
```

### Anomaly Severity Levels

| Z-Score | Severity | Score |
|---------|----------|-------|
| > 4.0   | Critical | 1.0   |
| > 3.0   | Warning  | 0.75  |
| > 2.0   | Watch    | 0.5   |
| ≤ 2.0   | Normal   | 0     |

## Command Line Options

```
usage: proteus_gateway.py [-h] [--dashboard] [--port PORT]
                          [--mode {learn,monitor,off}] [--baseline BASELINE]

Options:
  --dashboard           Start local NiceGUI dashboard
  --port PORT           Dashboard port (default: 8080)
  --mode MODE           Anomaly detection mode:
                        - learn: Collect baseline from healthy machine
                        - monitor: Detect anomalies using baseline
                        - off: Disable anomaly detection (default)
  --baseline FILE       Baseline file path (default: baseline.json)
```

## Sensor Data

### ML Features (Pre-processed by Firmware)

| Feature | UUID | Description |
|---------|------|-------------|
| FFT Amplitude | 0x0005 | 20-byte frequency spectrum packets |
| Accel RMS | 0x0006 | RMS acceleration (mg) |
| Accel Peak | 0x0007 | Peak acceleration (mg) |
| Velocity RMS | 0x0008 | RMS velocity (mm/s) - **key metric** |
| ML Status | 0x0009 | Threshold status |

### Calculated Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| Velocity RMS | Vibration intensity | < 1.8 mm/s (ISO Zone A) |
| Crest Factor | Peak/RMS ratio | < 3.0 |
| Kurtosis | Signal "peakedness" | ~3.0 (Gaussian) |
| Bearing Health Index | Composite score | 80-100 |

### ISO 20816 Zones

| Zone | Velocity (mm/s) | Status |
|------|-----------------|--------|
| A | < 1.8 | Excellent |
| B | 1.8 - 4.5 | Good |
| C | 4.5 - 11.2 | Alert |
| D | > 11.2 | Danger |

## Project Structure

```
proteus-gateway/
├── proteus_gateway.py    # Main gateway (BLE, parsing, anomaly detection)
├── dashboard.py          # NiceGUI real-time dashboard
├── quick_test.py         # Simple connection test script
├── requirements.txt      # Python dependencies
├── .env.example          # Configuration template
├── baseline.json         # Learned anomaly baseline (generated)
└── README.md             # This file
```

## Configuration

Environment variables (set in `.env` or shell):

```bash
# Device
PROTEUS_DEVICE=PROTEUS
PROTEUS_ADDRESS=2402245D-06F8-8C06-6450-9C102D4D7CE6

# MQTT (ThingsBoard)
MQTT_ENABLED=true
MQTT_HOST=your-thingsboard-host
MQTT_PORT=1883
MQTT_USER=your-device-token

# NATS (optional)
NATS_ENABLED=false
NATS_URL=nats://localhost:4222
```

## Troubleshooting

### Device Not Found

1. Power cycle PROTEUS (unplug/replug)
2. Reset Bluetooth on host
3. Ensure not connected to another app (ST BLE Sensor app, etc.)

### No Velocity Data

The PROTEUS firmware only sends velocity RMS when vibration exceeds a threshold. Tap or shake the board to trigger data.

### Connection Drops

Normal for BLE. The gateway auto-reconnects with exponential backoff (max 2 minutes between attempts).

## Dependencies

```
bleak>=0.21.0      # BLE communication
paho-mqtt>=2.0.0   # MQTT client
nats-py>=2.0.0     # NATS client (optional)
nicegui>=2.0       # Dashboard (optional)
```

## License

MIT License - See LICENSE file

## References

- [STEVAL-PROTEUS1 Product Page](https://www.st.com/en/evaluation-tools/steval-proteus1.html)
- [BlueSTSDK Protocol](https://github.com/STMicroelectronics/BlueSTSDK)
- [ISO 20816 Vibration Standards](https://www.iso.org/standard/63180.html)
