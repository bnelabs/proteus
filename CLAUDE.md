# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cross-platform Python BLE gateway for the ST STEVAL-PROTEUS1 industrial vibration sensor. Connects via Bluetooth Low Energy, streams sensor data, performs vibration health analysis, and detects anomalies using statistical baselines. Supports Windows, macOS, and Linux.

## Common Commands

### Environment Setup

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Configure credentials (required)
cp .env.example .env
# Edit .env with your MQTT_HOST and MQTT_USER
```

### Running the Gateway

```bash
# Scan for PROTEUS devices (required on Windows/Linux to find MAC address)
python proteus_gateway.py --scan

# Run gateway (macOS usually auto-detects)
python proteus_gateway.py

# Run with specific device address (Windows/Linux)
python proteus_gateway.py --address AA:BB:CC:DD:EE:FF

# Learn baseline from healthy machine (anomaly detection)
python proteus_gateway.py --mode learn --baseline baseline.json

# Monitor for anomalies using baseline
python proteus_gateway.py --mode monitor --baseline baseline.json

# Run with dashboard
python proteus_gateway.py --dashboard --port 8080
```

### Testing

```bash
# Quick connection test (outputs to test_output.txt)
python quick_test.py
python quick_test.py --scan
python quick_test.py --address XX:XX:XX:XX:XX:XX

# Run standalone dashboard (connects to gateway automatically)
python dashboard.py
```

## Architecture

### Core Components

**proteus_gateway.py** - Main application with three primary classes:

1. **ProteusGateway** - Main orchestrator class
   - BLE connection management with exponential backoff reconnection
   - Multi-broker publishing (MQTT + NATS)
   - Data buffering during disconnections
   - Platform detection (macOS UUID vs Windows/Linux MAC address)
   - Manages sensor state aggregation for ThingsBoard telemetry

2. **AnomalyDetector** - Statistical anomaly detection engine
   - Two modes: LEARN (collect baseline) and MONITOR (detect deviations)
   - Tracks 5 features: velocity_rms, crest_factor, kurtosis, temperature, bearing_health_index
   - Uses z-scores with configurable thresholds (2.0=watch, 3.0=warning, 4.0=critical)
   - Baseline stored as JSON with mean, std, percentiles
   - Minimum 500 samples (~8 minutes) before baseline is valid

3. **SensorReading** - Data container class
   - Immutable dataclass representing single sensor reading
   - Includes sensor type, timestamp, device name, parsed data, raw bytes
   - Serializable to dict/JSON

**dashboard.py** - NiceGUI-based real-time visualization
   - Mobile-first responsive corporate UI
   - Real-time charts (vibration, FFT spectrum, temperature)
   - ISO 10816-1 zone classification display
   - Data export (CSV/JSON)
   - Help dialogs with industrial vibration standards reference

**quick_test.py** - Minimal connection test utility
   - Cross-platform device scanning
   - Basic BLE characteristic reading
   - Outputs to both console and test_output.txt file

### Data Flow

```
PROTEUS Device (BLE)
  ↓ Bleak BLE notifications
ProteusGateway._on_notification()
  ↓ Parse raw bytes using PARSERS dispatch table
SensorReading object
  ↓ Calculate vibration metrics (ISO zone, crest factor, kurtosis)
  ↓ Optional: AnomalyDetector.check()
  ↓ Aggregate to sensor_state dict
Publish to MQTT/NATS
  ↓ ThingsBoard telemetry format
Dashboard updates (if running)
```

### BlueST Protocol

The gateway implements ST's BlueST protocol for BLE sensor data:

- **Service UUID**: `00000000-0001-11e1-9ab4-0002a5d5c51b`
- **Feature Characteristic Base**: `00000000-0001-11e1-ac36-0002a5d5c51b`

**Key Characteristics** (0x0002 service - ML features):
- `0x0005`: FFT amplitude (20-byte packets, 256 frequency bins)
- `0x0006`: Acceleration RMS (X,Y,Z in g)
- `0x0007`: Acceleration Peak (X,Y,Z in g)
- `0x0008`: Velocity RMS (X,Y,Z in mm/s) - **primary health metric**
- `0x0009`: ML status/thresholds

**Parser Dispatch**: `PARSERS` dict maps sensor type string → parse function. Each parser handles struct unpacking of little-endian binary data.

### Platform-Specific Behavior

**Address Format Detection** (line 40-48):
- macOS: UUID format (e.g., `2402245D-06F8-8C06-6450-9C102D4D7CE6`)
- Windows/Linux: MAC address format (e.g., `AA:BB:CC:DD:EE:FF`)
- Platform detected via `platform.system()` → stored in `PLATFORM` constant

**Auto-Discovery**: If `device_address` is empty, gateway automatically scans for devices named "PROTEUS" or "cingoz" (configurable via `device_names` list).

### Vibration Analysis Calculations

**ISO 20816 Zone Classification** (`classify_iso_zone()` function):
- Zone A: < 1.8 mm/s (Excellent)
- Zone B: 1.8-4.5 mm/s (Good)
- Zone C: 4.5-11.2 mm/s (Alert)
- Zone D: > 11.2 mm/s (Danger)

**Crest Factor**: `peak_acceleration / rms_acceleration`
- Normal: < 3.0
- Early wear: 3.0-4.5
- Fault condition: > 4.5

**Kurtosis**: Statistical measure of signal "peakedness"
- Gaussian noise: ~3.0
- Developing fault: 4.0-6.0
- Active damage: > 6.0

**Bearing Health Index**: Composite 0-100 score from vibration metrics, crest factor, and kurtosis.

### Configuration

Configuration via `.env` file (auto-loaded by python-dotenv at startup):

```bash
cp .env.example .env
# Edit .env with your credentials
python proteus_gateway.py  # Auto-loads .env
```

**Required** (gateway fails without these):
- `MQTT_HOST` - MQTT broker hostname/IP
- `MQTT_USER` - ThingsBoard device access token

**Optional**:
- **Device**: `PROTEUS_DEVICE`, `PROTEUS_ADDRESS`, `PROTEUS_AUTO_SCAN`, `RECONNECT_MAX_DELAY`, `BUFFER_SIZE`, `HEALTH_CHECK_INTERVAL`
- **MQTT**: `MQTT_ENABLED`, `MQTT_PORT`, `MQTT_USE_TLS`, `MQTT_PASSWORD`, `MQTT_TOPIC`
- **NATS**: `NATS_ENABLED`, `NATS_URL`

Loaded in `CONFIG` dict at module level. Validation in `setup_mqtt()` ensures required vars are set.

### Reconnection Strategy

Exponential backoff implemented in `ProteusGateway.run()`:
- Initial delay: 1 second
- Max delay: 120 seconds (configurable via `RECONNECT_MAX_DELAY`)
- Multiplier: 2x per failed attempt
- Resets to 1 second after successful connection

Data buffered in `deque` (max 1000 items) during disconnections, published on reconnect.

### Anomaly Detection Details

**Learning Mode**:
- Collects samples for each tracked feature
- After MIN_SAMPLES (500), calculates mean, std, min, max, p5, p95
- Auto-saves baseline on Ctrl+C or when minimum reached
- Progress logged every 100 samples

**Monitoring Mode**:
- Loads baseline from JSON file
- For each new reading, calculates z-score: `(value - mean) / std`
- Flags anomaly if any feature exceeds threshold
- Returns: `{"anomaly": bool, "score": float, "reasons": [features], "severities": {...}}`

**Severity Mapping**:
- z > 4.0: Critical (score=1.0)
- z > 3.0: Warning (score=0.75)
- z > 2.0: Watch (score=0.5)
- z ≤ 2.0: Normal (score=0)

## Debug Scripts

The root directory contains multiple debug/test scripts (untracked by git):

- `activate_ml_features.py` - Enable ML features via BLE commands
- `debug_fft_aggressive.py` - FFT data streaming tests
- `debug_iis3dwb.py` - IIS3DWB accelerometer debugging
- `debug_p2p_command.py` - P2P characteristic command testing
- `debug_proteus.py` - General PROTEUS debugging
- `debug_temperature.py` - Temperature sensor tests
- `test_*.py` - Various BLE streaming and command tests

These are development tools for reverse-engineering the BlueST protocol and testing individual characteristics.

## Important Notes

### BLE Connection Stability

BLE connections are inherently unstable - the gateway expects and handles frequent disconnections. Auto-reconnect is normal behavior, not a bug.

### Velocity RMS Data Availability

The PROTEUS firmware only sends velocity RMS when vibration exceeds an internal threshold. To trigger data flow, physically tap or shake the board. This is by design, not a connection issue.

### ThingsBoard MQTT Format

Gateway publishes aggregated telemetry as single JSON payload to `v1/devices/me/telemetry` topic:
```json
{
  "accel_x": 0.5, "accel_y": 0.3, "accel_z": 9.8,
  "velocity_rms": 1.2,
  "iso_zone": "A",
  "temperature": 24.5,
  "anomaly": false
}
```

### Adding New Sensors

To add support for a new BLE characteristic:

1. Add UUID → name mapping to `KNOWN_CHARACTERISTICS` dict
2. Create parse function (signature: `bytes → dict`)
3. Add parser to `PARSERS` dispatch table
4. Parser should handle `len(data)` validation and struct unpacking
5. Return dict with semantic field names (not raw hex)
