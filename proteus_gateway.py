#!/usr/bin/env python3
"""
PROTEUS BLE Gateway - Improved Implementation

Features:
- Reconnection with exponential backoff
- Connection health monitoring
- Data buffering during disconnects
- Dual broker support (NATS + MQTT over WebSocket/TLS)
- Configuration via environment variables
- Graceful shutdown
- Structured logging
"""

import asyncio
import json
import logging
import math
import os
import signal
import struct
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

# Optional imports
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None

try:
    import nats
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False
    nats = None

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Device - search for PROTEUS or cingoz (local name varies)
    "device_name": os.getenv("PROTEUS_DEVICE", "PROTEUS"),
    "device_names": ["PROTEUS", "cingoz"],  # Multiple names to search for
    "device_address": os.getenv("PROTEUS_ADDRESS", "2402245D-06F8-8C06-6450-9C102D4D7CE6"),
    "reconnect_max_delay": int(os.getenv("RECONNECT_MAX_DELAY", "120")),
    "buffer_size": int(os.getenv("BUFFER_SIZE", "1000")),
    "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "5")),

    # NATS (optional)
    "nats_enabled": os.getenv("NATS_ENABLED", "false").lower() == "true",
    "nats_url": os.getenv("NATS_URL", "nats://localhost:4222"),

    # ThingsBoard MQTT Connection (direct IP to bypass Cloudflare)
    "mqtt_enabled": os.getenv("MQTT_ENABLED", "true").lower() == "true",
    "mqtt_host": os.getenv("MQTT_HOST", "65.21.194.174"),
    "mqtt_port": int(os.getenv("MQTT_PORT", "31883")),
    "mqtt_use_tls": os.getenv("MQTT_USE_TLS", "false").lower() == "true",
    "mqtt_user": os.getenv("MQTT_USER", "Pm9S0PyvImuyXye7sPB9"),  # Device access token
    "mqtt_pass": os.getenv("MQTT_PASSWORD", ""),  # Empty for ThingsBoard

    # ThingsBoard telemetry topic
    "mqtt_topic": os.getenv("MQTT_TOPIC", "v1/devices/me/telemetry"),
}

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("proteus_gateway")

# =============================================================================
# BlueST Protocol Constants
# =============================================================================

# Service UUID for BlueST features
BLUEST_SERVICE_UUID = "00000000-0001-11e1-9ab4-0002a5d5c51b"

# Feature characteristic base UUID (mask with feature bits)
FEATURE_CHAR_BASE = "00000000-0001-11e1-ac36-0002a5d5c51b"

# Feature masks (first 4 bytes of characteristic UUID)
# Reference: BlueSTSDK feature mask encoding
FEATURE_MASKS = {
    0x20000000: "fft_amplitude",      # FFT Amplitude data
    0x00000400: "accelerometer",      # ISM330DHCX accelerometer
    0x00000800: "gyroscope",          # ISM330DHCX gyroscope
    0x00020000: "temperature",        # STTS22H temperature
    0x00040000: "pressure",           # LPS22HH pressure
    0x00080000: "humidity",           # Humidity sensor
    0x00100000: "magnetometer",       # NOT on PROTEUS
    0x00200000: "mag_calibration",    # Magnetometer calibration
    0x00c00000: "gyroscope",          # Battery/Gyro (PROTEUS uses for gyro)
}

# Known BlueST characteristic UUIDs for PROTEUS
KNOWN_CHARACTERISTICS = {
    # Standard BlueST features
    "20000000-0001-11e1-ac36-0002a5d5c51b": "fft_amplitude_old",  # Only returns 3-byte ack
    "00000400-0001-11e1-ac36-0002a5d5c51b": "accelerometer",
    "00000800-0001-11e1-ac36-0002a5d5c51b": "gyroscope",
    "00020000-0001-11e1-ac36-0002a5d5c51b": "temperature",
    "00040000-0001-11e1-ac36-0002a5d5c51b": "pressure",
    "00080000-0001-11e1-ac36-0002a5d5c51b": "humidity",
    "001c0000-0001-11e1-ac36-0002a5d5c51b": "environmental",
    "00e00000-0001-11e1-ac36-0002a5d5c51b": "motion",
    "00c00000-0001-11e1-ac36-0002a5d5c51b": "gyroscope",  # PROTEUS gyro data

    # Extended features (0002 service - ML/AI outputs)
    # ML_Feature_5 is the REAL FFT data source! (discovered via iOS packet capture)
    "00000005-0002-11e1-ac36-0002a5d5c51b": "fft_amplitude",   # Real FFT - 20 byte packets!
    "00000006-0002-11e1-ac36-0002a5d5c51b": "accel_rms",       # Acceleration RMS (X,Y,Z) in g
    "00000007-0002-11e1-ac36-0002a5d5c51b": "accel_peak",      # Acceleration Peak (X,Y,Z)
    "00000008-0002-11e1-ac36-0002a5d5c51b": "velocity_rms",    # Velocity RMS
    "00000009-0002-11e1-ac36-0002a5d5c51b": "ml_status",       # ML Status/thresholds
    "00000014-0002-11e1-ac36-0002a5d5c51b": "ml_classification", # AI classification

    # P2P/Configuration characteristics
    "0000fe42-8e22-4541-9d4c-21edae82ed19": "p2p_notify",
    "0000fe41-8e22-4541-9d4c-21edae82ed19": "p2p_write",
    "0000fe11-8e22-4541-9d4c-21edae82ed19": "p2p_command",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SensorReading:
    """Represents a single sensor reading."""
    sensor_type: str
    timestamp: datetime
    device_name: str
    data: dict
    raw_bytes: bytes = field(default_factory=bytes, repr=False)

    def to_dict(self) -> dict:
        return {
            "type": self.sensor_type,
            "timestamp": self.timestamp.isoformat(),
            "device": self.device_name,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# =============================================================================
# Data Parsers
# =============================================================================

def parse_accelerometer(data: bytes) -> dict:
    """Parse accelerometer data from BlueST format."""
    if len(data) < 8:
        return {"error": "insufficient data", "raw": data.hex()}

    # Skip timestamp (2 bytes), then read X, Y, Z as int16
    timestamp = struct.unpack_from('<H', data, 0)[0]
    x = struct.unpack_from('<h', data, 2)[0]
    y = struct.unpack_from('<h', data, 4)[0]
    z = struct.unpack_from('<h', data, 6)[0]

    return {
        "packet_timestamp": timestamp,
        "x_mg": x,
        "y_mg": y,
        "z_mg": z,
    }


def parse_temperature(data: bytes) -> dict:
    """Parse temperature data from PROTEUS.

    Individual temperature characteristic (0x00020000) format:
    - 9 bytes: [timestamp 2B] [unknown 5B] [temp 2B]
    - Temperature at bytes 7-8, raw / 50.0 gives °C

    Note: The Environmental Combined characteristic (0x001c0000) uses
    different format with raw / 10.0, but that's handled by parse_environmental().
    """
    if len(data) < 4:
        return {"error": "insufficient data", "raw": data.hex()}

    timestamp = struct.unpack_from('<H', data, 0)[0]

    # Handle 9-byte PROTEUS format (individual temperature characteristic)
    if len(data) >= 9:
        # Temperature is at bytes 7-8 in little endian
        # Using /50.0 scale based on calibration (raw=1152 → ~23°C)
        temp_raw = struct.unpack_from('<H', data, 7)[0]
        temp_c = temp_raw / 50.0
        return {
            "packet_timestamp": timestamp,
            "temperature_c": temp_c,
            "temperature_raw": temp_raw,
        }

    # Fallback for standard 4-byte format
    temp = struct.unpack_from('<h', data, 2)[0] / 10.0

    return {
        "packet_timestamp": timestamp,
        "temperature_c": temp,
    }


def parse_environmental(data: bytes) -> dict:
    """Parse combined environmental data (pressure, humidity, temperature)."""
    if len(data) < 10:
        return {"error": "insufficient data", "raw": data.hex()}

    timestamp = struct.unpack_from('<H', data, 0)[0]
    pressure = struct.unpack_from('<I', data, 2)[0] / 100.0  # mbar
    humidity = struct.unpack_from('<h', data, 6)[0] / 10.0  # %
    temp = struct.unpack_from('<h', data, 8)[0] / 10.0  # Celsius

    return {
        "packet_timestamp": timestamp,
        "pressure_mbar": pressure,
        "humidity_percent": humidity,
        "temperature_c": temp,
    }


def parse_motion(data: bytes) -> dict:
    """Parse combined motion data (accelerometer, gyroscope, magnetometer)."""
    result = {"packet_timestamp": 0}

    if len(data) < 2:
        return {"error": "insufficient data", "raw": data.hex()}

    result["packet_timestamp"] = struct.unpack_from('<H', data, 0)[0]
    offset = 2

    # Accelerometer (6 bytes: X, Y, Z as int16)
    if len(data) >= offset + 6:
        result["accel_x_mg"] = struct.unpack_from('<h', data, offset)[0]
        result["accel_y_mg"] = struct.unpack_from('<h', data, offset + 2)[0]
        result["accel_z_mg"] = struct.unpack_from('<h', data, offset + 4)[0]
        offset += 6

    # Gyroscope (6 bytes: X, Y, Z as int16)
    if len(data) >= offset + 6:
        result["gyro_x_dps"] = struct.unpack_from('<h', data, offset)[0] / 10.0
        result["gyro_y_dps"] = struct.unpack_from('<h', data, offset + 2)[0] / 10.0
        result["gyro_z_dps"] = struct.unpack_from('<h', data, offset + 4)[0] / 10.0
        offset += 6

    # Magnetometer (6 bytes: X, Y, Z as int16)
    if len(data) >= offset + 6:
        result["mag_x_mgauss"] = struct.unpack_from('<h', data, offset)[0]
        result["mag_y_mgauss"] = struct.unpack_from('<h', data, offset + 2)[0]
        result["mag_z_mgauss"] = struct.unpack_from('<h', data, offset + 4)[0]

    return result


def parse_fft_amplitude(data: bytes) -> dict:
    """Parse FFT Amplitude characteristic (0x0500).

    Firmware FFT Payload Header format:
    - Offset 0-1:  Magnitude size (16-bit LE, number of bins)
    - Offset 2:    Number of axes (1 byte, always 3)
    - Offset 3-6:  Bin frequency step (32-bit LE float)
    - Offset 7+:   Float32 magnitudes [X then Y then Z]

    FFT Configuration from firmware:
    - TACQ_DEFAULT = 2000ms (2 seconds acquisition time)
    - FFT_SIZE_DEFAULT = 1024 samples
    """
    if len(data) < 7:
        return {"raw_hex": data.hex(), "length": len(data)}

    # For old 3-byte ack packets (20000000-0001)
    if len(data) <= 4:
        return {"raw_hex": data.hex(), "length": len(data), "type": "ack"}

    # Header packet format per firmware
    mag_size = struct.unpack_from('<H', data, 0)[0]
    num_axes = data[2]
    bin_freq = struct.unpack_from('<f', data, 3)[0]

    # Extract magnitudes starting at offset 7
    amplitudes = []
    for i in range((len(data) - 7) // 4):
        try:
            val = struct.unpack_from('<f', data, 7 + i * 4)[0]
            amplitudes.append(round(val, 6))
        except:
            pass

    return {
        "magnitude_size": mag_size,
        "num_axes": num_axes,
        "bin_frequency_hz": round(bin_freq, 2),
        "amplitudes": amplitudes,
        "num_samples": len(amplitudes),
    }


def parse_pressure(data: bytes) -> dict:
    """Parse pressure data from PROTEUS.

    Individual pressure characteristic (0x00040000) format:
    - 4 bytes: [timestamp 2B] [pressure 2B]
    - Pressure raw + 700 gives mbar

    Note: The Environmental Combined characteristic (0x001c0000) uses
    32-bit pressure with /100.0, but that's handled by parse_environmental().
    """
    if len(data) < 4:
        return {"error": "insufficient data", "raw": data.hex()}

    timestamp = struct.unpack_from('<H', data, 0)[0]
    pressure_raw = struct.unpack_from('<H', data, 2)[0]

    # PROTEUS individual pressure characteristic needs offset of ~700 to convert to mbar
    pressure_mbar = pressure_raw + 700.0

    return {
        "packet_timestamp": timestamp,
        "pressure_mbar": pressure_mbar,
        "pressure_raw": pressure_raw,
    }


def parse_gyro_mag(data: bytes) -> dict:
    """Parse gyroscope data (14 bytes) from PROTEUS.

    Format: [timestamp 2B] [gyro_x 2B] [gyro_y 2B] [gyro_z 2B] [extra1 2B] [extra2 2B] [extra3 2B]

    Note: PROTEUS does NOT have a magnetometer. The last 3 values are likely
    gyro bias or other sensor data, not magnetometer readings.

    Gyro values are in 0.1 dps (tenths of degrees per second), divide by 10 for dps.
    """
    if len(data) < 8:
        return {"error": "insufficient data", "raw": data.hex()}

    timestamp = struct.unpack_from('<H', data, 0)[0]

    # Gyroscope X, Y, Z as int16 (in 0.1 dps)
    gyro_x = struct.unpack_from('<h', data, 2)[0]
    gyro_y = struct.unpack_from('<h', data, 4)[0]
    gyro_z = struct.unpack_from('<h', data, 6)[0]

    result = {
        "packet_timestamp": timestamp,
        "gyro_x_raw": gyro_x,
        "gyro_y_raw": gyro_y,
        "gyro_z_raw": gyro_z,
        "gyro_x_dps": gyro_x / 10.0,
        "gyro_y_dps": gyro_y / 10.0,
        "gyro_z_dps": gyro_z / 10.0,
    }

    # Extra values (bytes 8-13) - not magnetometer on PROTEUS
    if len(data) >= 14:
        extra1 = struct.unpack_from('<h', data, 8)[0]
        extra2 = struct.unpack_from('<h', data, 10)[0]
        extra3 = struct.unpack_from('<h', data, 12)[0]
        result["extra1"] = extra1
        result["extra2"] = extra2
        result["extra3"] = extra3

    return result


def parse_p2p_notify(data: bytes) -> dict:
    """Parse P2P/Inugo notify data (typically 5 bytes)."""
    if len(data) < 2:
        return {"raw_hex": data.hex(), "length": len(data)}

    timestamp = struct.unpack_from('<H', data, 0)[0]

    # Remaining bytes as status/data
    status_bytes = data[2:]

    return {
        "packet_timestamp": timestamp,
        "status": status_bytes.hex(),
        "length": len(data),
    }


def parse_generic(data: bytes) -> dict:
    """Generic parser for unknown characteristic data."""
    return {
        "raw_hex": data.hex(),
        "length": len(data),
    }


def parse_accel_rms(data: bytes) -> dict:
    """Parse ML Feature 6: Acceleration RMS (X, Y, Z).

    Format (20 bytes):
    - Bytes 0-3: Header/timestamp
    - Bytes 4-5: Counter or ID
    - Bytes 6-7: Counter or ID
    - Bytes 8-11: Accel RMS X (float32, in g)
    - Bytes 12-15: Accel RMS Y (float32, in g)
    - Bytes 16-19: Accel RMS Z (float32, in g)
    """
    if len(data) < 20:
        return {"raw_hex": data.hex(), "length": len(data)}

    header = struct.unpack_from('<I', data, 0)[0]

    # Acceleration RMS values in g (float32)
    accel_x = struct.unpack_from('<f', data, 8)[0]
    accel_y = struct.unpack_from('<f', data, 12)[0]
    accel_z = struct.unpack_from('<f', data, 16)[0]

    # Convert g to mg for consistency
    return {
        "header": header,
        "accel_rms_x_g": round(accel_x, 6),
        "accel_rms_y_g": round(accel_y, 6),
        "accel_rms_z_g": round(accel_z, 6),
        "accel_rms_x_mg": round(accel_x * 1000, 2),
        "accel_rms_y_mg": round(accel_y * 1000, 2),
        "accel_rms_z_mg": round(accel_z * 1000, 2),
    }


def parse_velocity_rms(data: bytes) -> dict:
    """Parse Time Domain characteristic (0x0600).

    Firmware format (20 bytes):
    - Offset 0-1:   Timestamp (16-bit LE)
    - Offset 2-3:   AccPeak X (16-bit LE, raw = value × 100)
    - Offset 4-5:   AccPeak Y (16-bit LE, raw = value × 100)
    - Offset 6-7:   AccPeak Z (16-bit LE, raw = value × 100)
    - Offset 8-11:  RMS Speed X (32-bit LE float, mm/s)
    - Offset 12-15: RMS Speed Y (32-bit LE float, mm/s)
    - Offset 16-19: RMS Speed Z (32-bit LE float, mm/s)

    Parsing:
    - AccPeak: raw / 100.0 → gives m/s²
    - RMS Speed: direct float → gives mm/s
    """
    if len(data) < 20:
        return {"raw_hex": data.hex(), "length": len(data)}

    timestamp = struct.unpack_from('<H', data, 0)[0]

    # AccPeak × 100 (16-bit LE signed)
    acc_peak_x = struct.unpack_from('<h', data, 2)[0] / 100.0
    acc_peak_y = struct.unpack_from('<h', data, 4)[0] / 100.0
    acc_peak_z = struct.unpack_from('<h', data, 6)[0] / 100.0

    # RMS Speed as float32 (mm/s)
    vel_x = struct.unpack_from('<f', data, 8)[0]
    vel_y = struct.unpack_from('<f', data, 12)[0]
    vel_z = struct.unpack_from('<f', data, 16)[0]

    # Calculate total RMS velocity
    vel_rms = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

    return {
        "packet_timestamp": timestamp,
        "acc_peak_x_ms2": round(acc_peak_x, 4),
        "acc_peak_y_ms2": round(acc_peak_y, 4),
        "acc_peak_z_ms2": round(acc_peak_z, 4),
        "velocity_x_mms": round(vel_x, 4),
        "velocity_y_mms": round(vel_y, 4),
        "velocity_z_mms": round(vel_z, 4),
        "velocity_rms_mms": round(vel_rms, 4),
    }


def parse_accel_peak(data: bytes) -> dict:
    """Parse ML Feature 7: Acceleration Peak (X, Y, Z).

    Format (15 bytes):
    - Bytes 0-2: Header/timestamp
    - Bytes 3-6: Accel Peak X (float32, in g)
    - Bytes 7-10: Accel Peak Y (float32, in g)
    - Bytes 11-14: Accel Peak Z (float32, in g)
    """
    if len(data) < 15:
        return {"raw_hex": data.hex(), "length": len(data)}

    # Acceleration Peak values in g (float32)
    accel_x = struct.unpack_from('<f', data, 3)[0]
    accel_y = struct.unpack_from('<f', data, 7)[0]
    accel_z = struct.unpack_from('<f', data, 11)[0]

    return {
        "accel_peak_x_g": round(accel_x, 6),
        "accel_peak_y_g": round(accel_y, 6),
        "accel_peak_z_g": round(accel_z, 6),
        "accel_peak_x_mg": round(accel_x * 1000, 2),
        "accel_peak_y_mg": round(accel_y * 1000, 2),
        "accel_peak_z_mg": round(accel_z * 1000, 2),
    }


def parse_ml_status(data: bytes) -> dict:
    """Parse ML Feature 9: Status and Thresholds.

    Format (15 bytes):
    - Bytes 0-2: Header/timestamp
    - Bytes 3-6: RMS threshold or status value
    - Bytes 7-10: Peak threshold or status value
    - Bytes 11-14: Warning/alarm status
    """
    if len(data) < 15:
        return {"raw_hex": data.hex(), "length": len(data)}

    # Parse as integers (could be threshold levels or status codes)
    val1 = struct.unpack_from('<I', data, 3)[0]
    val2 = struct.unpack_from('<I', data, 7)[0]
    val3 = struct.unpack_from('<I', data, 11)[0]

    return {
        "threshold_rms": val1,
        "threshold_peak": val2,
        "status": val3,
        "raw_hex": data.hex(),
    }


# Parser dispatch table
PARSERS = {
    "accelerometer": parse_accelerometer,
    "temperature": parse_temperature,
    "environmental": parse_environmental,
    "motion": parse_motion,
    "p2p_notify": parse_p2p_notify,
    "fft_amplitude": parse_fft_amplitude,
    "pressure": parse_pressure,
    "gyro_mag": parse_gyro_mag,
    "gyroscope": parse_gyro_mag,  # Same parser as gyro_mag
    "accel_rms": parse_accel_rms,  # ML Feature 6 - Real accel data!
    "accel_peak": parse_accel_peak,  # ML Feature 7 - Peak acceleration
    "velocity_rms": parse_velocity_rms,  # ML Feature 8 - Pre-calculated velocity!
    "ml_status": parse_ml_status,  # ML Feature 9 - Status/thresholds
}


def identify_characteristic(uuid: str) -> str:
    """Identify the sensor type from characteristic UUID."""
    uuid_lower = uuid.lower()
    if uuid_lower in KNOWN_CHARACTERISTICS:
        return KNOWN_CHARACTERISTICS[uuid_lower]

    # Try to extract feature mask from UUID
    try:
        feature_hex = uuid_lower.split('-')[0]
        feature_mask = int(feature_hex, 16)
        for mask, name in FEATURE_MASKS.items():
            if feature_mask & mask:
                return name
    except (ValueError, IndexError):
        pass

    return "unknown"


# =============================================================================
# Velocity Calculation & ISO Zone Classification
# =============================================================================

def accel_to_velocity(accel_mg: float, sample_rate_hz: float = 26.0) -> float:
    """
    Convert acceleration (mg) to velocity (mm/s).

    Uses simple integration: v = a * dt
    For vibration analysis, this is an approximation.
    accel_mg: acceleration in milli-g
    sample_rate_hz: sampling frequency
    Returns velocity in mm/s
    """
    # Convert mg to m/s² (1g = 9.81 m/s², 1mg = 0.00981 m/s²)
    accel_ms2 = accel_mg * 0.00981

    # Integrate over sample period: v = a * dt
    dt = 1.0 / sample_rate_hz
    velocity_ms = accel_ms2 * dt

    # Convert to mm/s
    velocity_mms = velocity_ms * 1000

    return velocity_mms


def calculate_velocity_rms(vx: float, vy: float, vz: float) -> float:
    """Calculate RMS velocity from X, Y, Z components."""
    return math.sqrt(vx**2 + vy**2 + vz**2)


def classify_iso_zone(velocity_rms_mms: float) -> str:
    """
    Classify vibration severity according to ISO 10816-1.

    Zones based on velocity RMS (mm/s):
    - Zone A: 0 - 1.12 mm/s (Good)
    - Zone B: 1.12 - 2.8 mm/s (Acceptable)
    - Zone C: 2.8 - 7.1 mm/s (Unsatisfactory)
    - Zone D: > 7.1 mm/s (Unacceptable)
    """
    if velocity_rms_mms <= 1.12:
        return "A"
    elif velocity_rms_mms <= 2.8:
        return "B"
    elif velocity_rms_mms <= 7.1:
        return "C"
    else:
        return "D"


# =============================================================================
# Statistical Bearing Health Functions
# =============================================================================

def calculate_crest_factor(peak: float, rms: float) -> float:
    """Calculate Crest Factor = Peak / RMS.

    Healthy bearing: < 3.0
    Warning: 3.0 - 4.5
    Critical: > 4.5
    """
    if rms <= 0:
        return 0.0
    return peak / rms


def calculate_kurtosis(samples: list) -> float:
    """Calculate Kurtosis (peakedness of distribution).

    Kurtosis = E[(X-μ)⁴] / σ⁴
    Gaussian distribution = 3.0
    Healthy bearing: ~3.0
    Impacting/degraded: > 4.0
    """
    if len(samples) < 4:
        return 3.0  # Default to Gaussian

    n = len(samples)
    mean = sum(samples) / n
    variance = sum((x - mean) ** 2 for x in samples) / n

    if variance <= 0:
        return 3.0

    std = math.sqrt(variance)
    fourth_moment = sum((x - mean) ** 4 for x in samples) / n
    kurtosis = fourth_moment / (std ** 4)

    return kurtosis


def calculate_bearing_health_index(crest_factor: float, kurtosis: float,
                                   velocity_rms: float, temperature_delta: float) -> float:
    """Calculate composite Bearing Health Index (0-100).

    Combines multiple indicators:
    - Crest Factor score (25%)
    - Kurtosis score (25%)
    - Velocity RMS / ISO zone score (35%)
    - Temperature score (15%)

    Returns: 0-100 (100 = healthy, 0 = critical)
    """
    # Crest Factor score (25 points max)
    if crest_factor < 3.0:
        cf_score = 25
    elif crest_factor < 4.5:
        cf_score = 25 - (crest_factor - 3.0) * (15 / 1.5)  # Linear decline
    else:
        cf_score = max(0, 10 - (crest_factor - 4.5) * 5)

    # Kurtosis score (25 points max)
    if kurtosis < 4.0:
        kurt_score = 25
    elif kurtosis < 6.0:
        kurt_score = 25 - (kurtosis - 4.0) * (15 / 2.0)
    else:
        kurt_score = max(0, 10 - (kurtosis - 6.0) * 5)

    # Velocity RMS score (35 points max) - ISO 10816-1
    if velocity_rms <= 1.12:      # Zone A
        vel_score = 35
    elif velocity_rms <= 2.8:     # Zone B
        vel_score = 28
    elif velocity_rms <= 7.1:     # Zone C
        vel_score = 15
    else:                         # Zone D
        vel_score = 0

    # Temperature delta score (15 points max)
    if temperature_delta < 10:
        temp_score = 15
    elif temperature_delta < 20:
        temp_score = 10
    elif temperature_delta < 30:
        temp_score = 5
    else:
        temp_score = 0

    return round(cf_score + kurt_score + vel_score + temp_score, 1)


def classify_health_status(health_index: float) -> str:
    """Classify health status from index."""
    if health_index >= 80:
        return "HEALTHY"
    elif health_index >= 50:
        return "WARNING"
    else:
        return "CRITICAL"


# =============================================================================
# Anomaly Detection
# =============================================================================

class AnomalyDetector:
    """
    Simple statistical anomaly detector for vibration monitoring.

    Modes:
    - LEARN: Collect baseline data from healthy machine operation
    - MONITOR: Compare real-time data against learned baseline

    Usage:
        detector = AnomalyDetector(mode="learn", baseline_file="baseline.json")
        detector.update(velocity_rms=0.85, crest_factor=2.7, temperature=24.5)
        # After enough samples, call detector.save_baseline()

        # Later, in monitor mode:
        detector = AnomalyDetector(mode="monitor", baseline_file="baseline.json")
        result = detector.check(velocity_rms=2.85, crest_factor=4.1, temperature=35.0)
        # result = {"anomaly": True, "score": 0.87, "reasons": [...]}
    """

    # Minimum samples before baseline is considered valid
    MIN_SAMPLES = 500  # ~8 minutes at 1 sample/sec

    # Features to track
    FEATURES = ["velocity_rms", "crest_factor", "kurtosis", "temperature", "bearing_health_index"]

    def __init__(self, mode: str = "learn", baseline_file: str = "baseline.json"):
        """
        Initialize anomaly detector.

        Args:
            mode: "learn" to collect baseline, "monitor" to detect anomalies
            baseline_file: Path to save/load baseline data
        """
        self.mode = mode.lower()
        self.baseline_file = baseline_file
        self.baseline: Dict[str, Dict[str, float]] = {}
        self.samples: Dict[str, list] = {f: [] for f in self.FEATURES}
        self.sample_count = 0
        self.baseline_ready = False

        if self.mode == "monitor":
            self._load_baseline()

        logger.info(f"AnomalyDetector initialized in {self.mode.upper()} mode")
        if self.mode == "monitor" and self.baseline_ready:
            logger.info(f"Baseline loaded from {baseline_file}")

    def _load_baseline(self) -> bool:
        """Load baseline from file."""
        try:
            with open(self.baseline_file, 'r') as f:
                data = json.load(f)
                self.baseline = data.get("baseline", {})
                self.sample_count = data.get("sample_count", 0)
                if self.baseline and all(f in self.baseline for f in self.FEATURES):
                    self.baseline_ready = True
                    logger.info(f"Baseline loaded: {self.sample_count} samples, "
                               f"velocity_rms mean={self.baseline['velocity_rms']['mean']:.4f}")
                    return True
        except FileNotFoundError:
            logger.warning(f"Baseline file not found: {self.baseline_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid baseline file: {e}")
        return False

    def save_baseline(self) -> bool:
        """Save current baseline to file."""
        if not self.baseline:
            self._calculate_baseline()

        if not self.baseline:
            logger.error("Cannot save - no baseline calculated")
            return False

        data = {
            "baseline": self.baseline,
            "sample_count": self.sample_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "min_samples": self.MIN_SAMPLES
        }

        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Baseline saved to {self.baseline_file}")
            return True
        except IOError as e:
            logger.error(f"Failed to save baseline: {e}")
            return False

    def _calculate_baseline(self):
        """Calculate baseline statistics from collected samples."""
        if self.sample_count < self.MIN_SAMPLES:
            logger.warning(f"Not enough samples: {self.sample_count}/{self.MIN_SAMPLES}")
            return

        for feature in self.FEATURES:
            values = self.samples[feature]
            if not values:
                continue

            # Filter out zeros and NaN
            valid_values = [v for v in values if v is not None and v > 0 and not math.isnan(v)]
            if len(valid_values) < 10:
                continue

            mean = sum(valid_values) / len(valid_values)
            variance = sum((x - mean) ** 2 for x in valid_values) / len(valid_values)
            std = math.sqrt(variance)

            # Calculate percentiles for robust bounds
            sorted_vals = sorted(valid_values)
            p5 = sorted_vals[int(len(sorted_vals) * 0.05)]
            p95 = sorted_vals[int(len(sorted_vals) * 0.95)]

            self.baseline[feature] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "min": round(min(valid_values), 6),
                "max": round(max(valid_values), 6),
                "p5": round(p5, 6),
                "p95": round(p95, 6),
                "count": len(valid_values)
            }

        self.baseline_ready = bool(self.baseline)
        logger.info(f"Baseline calculated from {self.sample_count} samples")

    def update(self, **kwargs):
        """
        Update with new sensor data (LEARN mode).

        Args:
            velocity_rms: Velocity RMS in mm/s
            crest_factor: Crest factor (peak/RMS)
            kurtosis: Signal kurtosis
            temperature: Temperature in °C
            bearing_health_index: Composite health score 0-100
        """
        if self.mode != "learn":
            return

        for feature in self.FEATURES:
            if feature in kwargs and kwargs[feature] is not None:
                self.samples[feature].append(kwargs[feature])

        self.sample_count += 1

        # Log progress periodically
        if self.sample_count % 100 == 0:
            progress = min(100, (self.sample_count / self.MIN_SAMPLES) * 100)
            logger.info(f"Learning progress: {self.sample_count} samples ({progress:.0f}%)")

        # Auto-calculate baseline when enough samples
        if self.sample_count == self.MIN_SAMPLES:
            logger.info("Minimum samples reached - calculating baseline...")
            self._calculate_baseline()
            self.save_baseline()

    def check(self, **kwargs) -> Dict[str, Any]:
        """
        Check for anomalies (MONITOR mode).

        Args:
            Same as update()

        Returns:
            {
                "anomaly": bool,
                "score": float (0-1, higher = more anomalous),
                "reasons": [{"feature": str, "value": float, "expected": str, "severity": str}]
            }
        """
        if self.mode != "monitor":
            return {"anomaly": False, "score": 0, "reasons": [], "status": "not_monitoring"}

        if not self.baseline_ready:
            return {"anomaly": False, "score": 0, "reasons": [], "status": "no_baseline"}

        reasons = []
        anomaly_scores = []

        for feature in self.FEATURES:
            if feature not in kwargs or kwargs[feature] is None:
                continue
            if feature not in self.baseline:
                continue

            value = kwargs[feature]
            stats = self.baseline[feature]
            mean = stats["mean"]
            std = stats["std"]

            # Skip if std is too small (avoid division by zero)
            if std < 0.0001:
                continue

            # Calculate z-score (how many standard deviations from mean)
            z_score = abs(value - mean) / std

            # Determine severity
            if z_score > 4:
                severity = "critical"
                anomaly_scores.append(1.0)
            elif z_score > 3:
                severity = "warning"
                anomaly_scores.append(0.75)
            elif z_score > 2:
                severity = "watch"
                anomaly_scores.append(0.5)
            else:
                continue  # Normal

            reasons.append({
                "feature": feature,
                "value": round(value, 4),
                "mean": round(mean, 4),
                "z_score": round(z_score, 2),
                "expected_range": f"{stats['p5']:.4f} - {stats['p95']:.4f}",
                "severity": severity
            })

        # Overall anomaly score (0-1)
        overall_score = max(anomaly_scores) if anomaly_scores else 0
        is_anomaly = overall_score >= 0.5  # Warning or above

        return {
            "anomaly": is_anomaly,
            "score": round(overall_score, 3),
            "reasons": reasons,
            "status": "monitoring"
        }

    def get_status(self) -> Dict[str, Any]:
        """Get detector status."""
        return {
            "mode": self.mode,
            "baseline_ready": self.baseline_ready,
            "sample_count": self.sample_count,
            "min_samples": self.MIN_SAMPLES,
            "progress": min(100, (self.sample_count / self.MIN_SAMPLES) * 100) if self.mode == "learn" else 100,
            "baseline_file": self.baseline_file
        }


# Global anomaly detector instance (set in main())
anomaly_detector: Optional[AnomalyDetector] = None


# =============================================================================
# PROTEUS Gateway
# =============================================================================

class ProteusGateway:
    """BLE Gateway for PROTEUS sensor board with auto-reconnect and broker support."""

    def __init__(self):
        self.device = None
        self.client: Optional[BleakClient] = None
        self.running = True
        self.connected = False

        # Data buffer for disconnection periods
        self.buffer: deque = deque(maxlen=CONFIG["buffer_size"])
        self.buffer_lock = asyncio.Lock()

        # Broker clients
        self.mqtt_client: Optional[mqtt.Client] = None
        self.nats_client = None

        # Track subscribed characteristics
        self.subscribed_chars: set = set()

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_buffered": 0,
            "reconnect_count": 0,
            "last_message_time": None,
        }

        # Sensor state for ThingsBoard telemetry aggregation
        self.sensor_state: Dict[str, Any] = {
            "accel_x": 0,
            "accel_y": 0,
            "accel_z": 0,
            "gyro_x": 0,
            "gyro_y": 0,
            "gyro_z": 0,
            "temperature": 0.0,
            "pressure": 0.0,
            "fft_amplitudes": [],  # FFT frequency bin amplitudes
            "last_update": None,
        }
        self.sensor_state_lock = asyncio.Lock()

        # Statistical analysis buffers
        self.accel_buffer: deque = deque(maxlen=256)  # ~2-4 seconds of data
        self.baseline_temperature: float = None  # Set on first reading

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False

    # =========================================================================
    # MQTT Setup
    # =========================================================================

    def setup_mqtt(self) -> bool:
        """Setup MQTT client for ThingsBoard (standard TCP or TLS)."""
        if not CONFIG["mqtt_enabled"]:
            logger.info("MQTT disabled via configuration")
            return True

        if not MQTT_AVAILABLE:
            logger.error("MQTT enabled but paho-mqtt not installed. Install with: pip install paho-mqtt")
            return False

        try:
            self.mqtt_client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"proteus-gateway-{CONFIG['device_name']}",
                transport="tcp",  # Standard TCP for ThingsBoard
                protocol=mqtt.MQTTv311,
            )

            # Enable TLS if configured (port 8883)
            if CONFIG["mqtt_use_tls"]:
                self.mqtt_client.tls_set()

            # Set credentials (ThingsBoard: username=access_token, password=empty)
            self.mqtt_client.username_pw_set(
                CONFIG["mqtt_user"],
                CONFIG["mqtt_pass"] or None
            )

            # Connection callbacks
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

            # Connect
            self.mqtt_client.connect(
                CONFIG["mqtt_host"],
                CONFIG["mqtt_port"],
                keepalive=60
            )

            # Start background thread for network loop
            self.mqtt_client.loop_start()

            protocol = "mqtts" if CONFIG["mqtt_use_tls"] else "mqtt"
            logger.info(f"MQTT connecting to {protocol}://{CONFIG['mqtt_host']}:{CONFIG['mqtt_port']}")
            return True

        except Exception as e:
            logger.error(f"Failed to setup MQTT: {e}")
            return False

    def _on_mqtt_connect(self, client, userdata, flags, reason_code, properties):
        """MQTT connection callback (API v2)."""
        if reason_code == 0:
            logger.info(f"Connected to ThingsBoard MQTT: {CONFIG['mqtt_host']}")
        else:
            logger.error(f"MQTT connection failed: {reason_code}")

    def _on_mqtt_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        """MQTT disconnection callback (API v2)."""
        if reason_code != 0:
            logger.warning(f"Unexpected MQTT disconnect ({reason_code}), will auto-reconnect")
        else:
            logger.info("MQTT disconnected cleanly")

    # =========================================================================
    # NATS Setup
    # =========================================================================

    async def setup_nats(self) -> bool:
        """Setup NATS client."""
        if not CONFIG["nats_enabled"]:
            logger.info("NATS disabled via configuration")
            return True

        if not NATS_AVAILABLE:
            logger.error("NATS enabled but nats-py not installed. Install with: pip install nats-py")
            return False

        try:
            self.nats_client = await nats.connect(
                servers=[CONFIG["nats_url"]],
                reconnect_time_wait=2,
                max_reconnect_attempts=-1,  # Unlimited
            )
            logger.info(f"Connected to NATS: {CONFIG['nats_url']}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            return False

    # =========================================================================
    # BLE Connection
    # =========================================================================

    async def scan_for_device(self) -> bool:
        """Scan for the PROTEUS device by name or known address."""
        search_names = CONFIG.get("device_names", [CONFIG["device_name"]])
        known_address = CONFIG.get("device_address", "").upper()

        logger.info(f"Scanning for device: {search_names} or address {known_address[:12]}...")

        try:
            devices = await BleakScanner.discover(timeout=10.0)

            for device in devices:
                device_name = device.name or ""
                device_addr = (device.address or "").upper()

                # Check by known address first
                if known_address and known_address in device_addr:
                    self.device = device
                    logger.info(f"Found device by address: {device.name or 'unnamed'} ({device.address})")
                    return True

                # Check by name (PROTEUS or cingoz)
                for search_name in search_names:
                    if search_name.lower() in device_name.lower():
                        self.device = device
                        logger.info(f"Found device: {device.name} ({device.address})")
                        return True

            logger.warning(f"Device not found. Searched for names={search_names}, address={known_address[:12]}...")
            return False

        except BleakError as e:
            logger.error(f"Scan failed: {e}")
            return False

        except Exception as e:
            logger.error(f"Scan error: {e}")
            if "Bluetooth" in str(e) or "permission" in str(e).lower():
                logger.error("On macOS, grant Bluetooth access to Terminal in:")
                logger.error("System Preferences > Privacy & Security > Bluetooth")
            return False

    async def connect_with_retry(self) -> bool:
        """Connect to the device with exponential backoff."""
        delay = 1
        known_address = CONFIG.get("device_address", "")

        while self.running:
            # First, make sure we have a device to connect to
            if self.device is None:
                if not await self.scan_for_device():
                    # If scan failed but we have a known address, try direct connection
                    if known_address:
                        logger.info(f"Scan failed, trying direct connect to {known_address[:12]}...")
                        try:
                            self.client = BleakClient(
                                known_address,
                                disconnected_callback=self._on_disconnect
                            )
                            await self.client.connect(timeout=30)
                            if self.client.is_connected:
                                self.connected = True
                                logger.info(f"Connected directly to {known_address}")
                                return True
                        except Exception as e:
                            logger.warning(f"Direct connection failed: {e}")

                    logger.warning(f"Device scan failed, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, CONFIG["reconnect_max_delay"])
                    continue

            try:
                # Create new client for each connection attempt
                self.client = BleakClient(
                    self.device.address,
                    disconnected_callback=self._on_disconnect
                )

                await self.client.connect(timeout=30)

                if self.client.is_connected:
                    self.connected = True
                    logger.info(f"Connected to {self.device.name} ({self.device.address})")
                    return True

            except BleakError as e:
                logger.warning(f"Connection failed: {e}")

            except Exception as e:
                logger.error(f"Unexpected error during connect: {e}")

            logger.info(f"Retrying connection in {delay}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, CONFIG["reconnect_max_delay"])
            self.stats["reconnect_count"] += 1

            # Re-scan after failed connection attempts
            self.device = None

        return False

    def _on_disconnect(self, client: BleakClient):
        """Callback when BLE device disconnects."""
        self.connected = False
        self.subscribed_chars.clear()
        logger.warning(f"Disconnected from {CONFIG['device_name']}")

    # =========================================================================
    # Continuous Streaming Enable (BlueSTSDK Protocol)
    # =========================================================================

    # BlueSTSDK feature masks for sensor enable commands
    # Note: PROTEUS does NOT have a humidity sensor
    SENSOR_FEATURE_MASKS = {
        "accelerometer": 0x00000400,
        "gyroscope": 0x00C00000,  # PROTEUS uses this mask for gyro
        "temperature": 0x00020000,
        "pressure": 0x00040000,
        "fft": 0x20000000,
        "velocity_rms": 0x00000008,  # Time Domain: velocity RMS + acc peak
    }

    # BlueSTSDK config characteristic for writing enable commands
    CONFIG_CHAR_UUID = "0000fe41-8e22-4541-9d4c-21edae82ed19"

    async def send_sensor_enable_command(self, name: str, feature_mask: int) -> bool:
        """
        Send BlueSTSDK enable command for a sensor.

        Correct command format per firmware analysis:
        [4-byte BE mask][Command char][Data byte]

        Example: Enable accelerometer (mask 0x00000400):
        struct.pack('>I', 0x00000400) + bytes([ord('m'), 0x01])
        Result: 00 00 04 00 6D 01

        Command 'm' (0x6D) with data=0x01 enables streaming.
        """
        if not self.client or not self.client.is_connected:
            return False

        # Format: [4-byte BE mask][command char 'm'][enable byte 0x01]
        command = struct.pack('>I', feature_mask) + bytes([ord('m'), 0x01])
        try:
            await self.client.write_gatt_char(
                self.CONFIG_CHAR_UUID,
                command,
                response=False
            )
            logger.info(f"Enabled {name} sensor (mask=0x{feature_mask:08X})")
            return True
        except BleakError as e:
            logger.warning(f"Failed to enable {name}: {e}")
            return False

    async def enable_continuous_streaming(self) -> bool:
        """
        Enable continuous streaming mode on PROTEUS board using BlueSTSDK protocol.

        BlueSTSDK sensors require explicit enable commands written to the config
        characteristic. Just subscribing to notifications is NOT enough for some sensors.

        Command format: [0x01][Feature mask: 4 bytes Little Endian]
        Written to config characteristic: 0000fe41-8e22-4541-9d4c-21edae82ed19
        """
        if not self.client or not self.client.is_connected:
            logger.error("Cannot enable streaming: not connected")
            return False

        logger.info("Sending BlueSTSDK sensor enable commands...")

        enabled_count = 0
        for name, mask in self.SENSOR_FEATURE_MASKS.items():
            if await self.send_sensor_enable_command(name, mask):
                enabled_count += 1
            # Small delay between commands to let board process
            await asyncio.sleep(0.1)

        if enabled_count > 0:
            logger.info(f"Enabled {enabled_count} sensors for continuous streaming")
        else:
            logger.warning("No sensors could be enabled - check device connection")

        return enabled_count > 0

    # =========================================================================
    # Characteristic Subscription
    # =========================================================================

    async def subscribe_all(self) -> bool:
        """Subscribe to all known BlueST characteristics."""
        if not self.client or not self.client.is_connected:
            logger.error("Cannot subscribe: not connected")
            return False

        try:
            services = self.client.services

            for service in services:
                for char in service.characteristics:
                    # Check if it's a notifiable characteristic
                    if "notify" in char.properties:
                        sensor_type = identify_characteristic(char.uuid)

                        if char.uuid not in self.subscribed_chars:
                            try:
                                await self.client.start_notify(
                                    char.uuid,
                                    lambda sender, data, uuid=char.uuid, stype=sensor_type:
                                        self._notification_handler(uuid, stype, data)
                                )
                                self.subscribed_chars.add(char.uuid)
                                logger.info(f"Subscribed to {sensor_type} ({char.uuid})")

                            except BleakError as e:
                                logger.warning(f"Failed to subscribe to {char.uuid}: {e}")

            if self.subscribed_chars:
                logger.info(f"Subscribed to {len(self.subscribed_chars)} characteristics")
                return True
            else:
                logger.warning("No characteristics subscribed")
                return False

        except Exception as e:
            logger.error(f"Error during subscription: {e}")
            return False

    def _notification_handler(self, uuid: str, sensor_type: str, data: bytes):
        """Handle incoming BLE notifications."""
        try:
            # Debug: log notification counts
            if not hasattr(self, '_notify_counts'):
                self._notify_counts = {}
            self._notify_counts[sensor_type] = self._notify_counts.get(sensor_type, 0) + 1
            if self._notify_counts[sensor_type] % 20 == 1:  # Log every 20th
                logger.info(f"[NOTIFY] type={sensor_type} count={self._notify_counts[sensor_type]}")

            # Parse the data
            parser = PARSERS.get(sensor_type, parse_generic)
            parsed_data = parser(data)

            # Debug log for pressure (log first 3)
            if sensor_type == "pressure" and not hasattr(self, '_pressure_logged'):
                self._pressure_logged = 0
            if sensor_type == "pressure" and self._pressure_logged < 3:
                logger.info(f"Pressure notification: raw={data.hex()}, parsed={parsed_data}")
                self._pressure_logged += 1

            # Update sensor state for ThingsBoard telemetry
            self._update_sensor_state(sensor_type, parsed_data)

            # Create reading
            reading = SensorReading(
                sensor_type=sensor_type,
                timestamp=datetime.now(timezone.utc),
                device_name=CONFIG["device_name"],
                data=parsed_data,
                raw_bytes=data
            )

            # Queue for publishing
            asyncio.create_task(self._queue_reading(reading))

        except Exception as e:
            logger.error(f"Error handling notification from {uuid}: {e}")

    async def _queue_reading(self, reading: SensorReading):
        """Queue a reading for publishing."""
        async with self.buffer_lock:
            self.buffer.append(reading)
            self.stats["messages_buffered"] += 1

    def _update_sensor_state(self, sensor_type: str, data: dict):
        """Update sensor state with latest readings for telemetry aggregation."""
        # Extract values based on sensor type
        if sensor_type == "accel_rms":
            # ML Feature 6: Real acceleration RMS data!
            self.sensor_state["accel_x"] = data.get("accel_rms_x_mg", 0)
            self.sensor_state["accel_y"] = data.get("accel_rms_y_mg", 0)
            self.sensor_state["accel_z"] = data.get("accel_rms_z_mg", 0)
            self.sensor_state["accel_rms_x_g"] = data.get("accel_rms_x_g", 0)
            self.sensor_state["accel_rms_y_g"] = data.get("accel_rms_y_g", 0)
            self.sensor_state["accel_rms_z_g"] = data.get("accel_rms_z_g", 0)

            # Buffer acceleration magnitude for statistical analysis
            accel_mag = math.sqrt(
                data.get("accel_rms_x_g", 0)**2 +
                data.get("accel_rms_y_g", 0)**2 +
                data.get("accel_rms_z_g", 0)**2
            )
            self.accel_buffer.append(accel_mag)

            # Calculate statistics when buffer has enough data
            if len(self.accel_buffer) >= 32:
                samples = list(self.accel_buffer)
                rms = math.sqrt(sum(x**2 for x in samples) / len(samples))
                peak = max(abs(x) for x in samples)

                self.sensor_state["crest_factor"] = calculate_crest_factor(peak, rms)
                self.sensor_state["kurtosis"] = calculate_kurtosis(samples)
        elif sensor_type == "velocity_rms":
            # Time Domain characteristic: Velocity RMS + AccPeak (pre-calculated by firmware!)
            self.sensor_state["velocity_x"] = data.get("velocity_x_mms", 0)
            self.sensor_state["velocity_y"] = data.get("velocity_y_mms", 0)
            self.sensor_state["velocity_z"] = data.get("velocity_z_mms", 0)
            vel_rms = data.get("velocity_rms_mms", 0)
            self.sensor_state["velocity_rms"] = vel_rms
            logger.info(f"[VEL_RMS] notification: vel_rms={vel_rms}")
            # AccPeak values from Time Domain characteristic (in m/s²)
            if "acc_peak_x_ms2" in data:
                self.sensor_state["acc_peak_x_ms2"] = data.get("acc_peak_x_ms2", 0)
                self.sensor_state["acc_peak_y_ms2"] = data.get("acc_peak_y_ms2", 0)
                self.sensor_state["acc_peak_z_ms2"] = data.get("acc_peak_z_ms2", 0)

            # Use velocity magnitude for statistical buffer (more reliable than raw accel)
            vel_mag = math.sqrt(
                data.get("velocity_x_mms", 0)**2 +
                data.get("velocity_y_mms", 0)**2 +
                data.get("velocity_z_mms", 0)**2
            )
            self.accel_buffer.append(vel_mag)

            # Calculate statistics when buffer has enough data
            if len(self.accel_buffer) >= 32:
                samples = list(self.accel_buffer)
                rms = math.sqrt(sum(x**2 for x in samples) / len(samples))
                peak = max(abs(x) for x in samples)

                self.sensor_state["crest_factor"] = calculate_crest_factor(peak, rms)
                self.sensor_state["kurtosis"] = calculate_kurtosis(samples)
        elif sensor_type == "accel_peak":
            # ML Feature 7: Peak acceleration
            self.sensor_state["accel_peak_x"] = data.get("accel_peak_x_mg", 0)
            self.sensor_state["accel_peak_y"] = data.get("accel_peak_y_mg", 0)
            self.sensor_state["accel_peak_z"] = data.get("accel_peak_z_mg", 0)
        elif sensor_type == "ml_status":
            # ML Feature 9: Status and thresholds
            self.sensor_state["threshold_rms"] = data.get("threshold_rms", 0)
            self.sensor_state["threshold_peak"] = data.get("threshold_peak", 0)
            self.sensor_state["ml_status"] = data.get("status", 0)
        elif sensor_type == "accelerometer":
            self.sensor_state["accel_x"] = data.get("x_mg", 0)
            self.sensor_state["accel_y"] = data.get("y_mg", 0)
            self.sensor_state["accel_z"] = data.get("z_mg", 0)

            # Buffer accelerometer magnitude for statistical analysis (fallback if accel_rms not available)
            accel_mag = math.sqrt(
                data.get("x_mg", 0)**2 +
                data.get("y_mg", 0)**2 +
                data.get("z_mg", 0)**2
            ) / 1000.0  # Convert mg to g
            self.accel_buffer.append(accel_mag)

            # Calculate statistics when buffer has enough data
            if len(self.accel_buffer) >= 32:
                samples = list(self.accel_buffer)
                rms = math.sqrt(sum(x**2 for x in samples) / len(samples))
                peak = max(abs(x) for x in samples)

                self.sensor_state["crest_factor"] = calculate_crest_factor(peak, rms)
                self.sensor_state["kurtosis"] = calculate_kurtosis(samples)
        elif sensor_type == "motion":
            self.sensor_state["accel_x"] = data.get("accel_x_mg", self.sensor_state["accel_x"])
            self.sensor_state["accel_y"] = data.get("accel_y_mg", self.sensor_state["accel_y"])
            self.sensor_state["accel_z"] = data.get("accel_z_mg", self.sensor_state["accel_z"])
            if "gyro_x_dps" in data:
                self.sensor_state["gyro_x"] = int(data.get("gyro_x_dps", 0) * 1000)  # Convert to mdps
                self.sensor_state["gyro_y"] = int(data.get("gyro_y_dps", 0) * 1000)
                self.sensor_state["gyro_z"] = int(data.get("gyro_z_dps", 0) * 1000)
        elif sensor_type in ("gyro_mag", "gyroscope"):
            # gyro_x_raw is in 0.1 dps, multiply by 100 to get mdps
            self.sensor_state["gyro_x"] = data.get("gyro_x_raw", 0) * 100
            self.sensor_state["gyro_y"] = data.get("gyro_y_raw", 0) * 100
            self.sensor_state["gyro_z"] = data.get("gyro_z_raw", 0) * 100
        elif sensor_type == "fft_amplitude":
            # Store FFT data if present (from ML_Feature_5, 20-byte packets)
            if "amplitudes" in data and data["amplitudes"]:
                # Append new amplitudes to buffer (keep last 64 values for analysis)
                current = self.sensor_state.get("fft_amplitudes", [])
                current.extend(data["amplitudes"])
                self.sensor_state["fft_amplitudes"] = current[-64:]  # Keep last 64 values
        elif sensor_type == "temperature":
            self.sensor_state["temperature"] = data.get("temperature_c", 0.0)
        elif sensor_type == "pressure":
            pressure = data.get("pressure_mbar", 0.0)
            if pressure > 0:
                self.sensor_state["pressure"] = pressure
        elif sensor_type == "environmental":
            self.sensor_state["temperature"] = data.get("temperature_c", self.sensor_state["temperature"])
            self.sensor_state["pressure"] = data.get("pressure_mbar", self.sensor_state["pressure"])

        self.sensor_state["last_update"] = datetime.now(timezone.utc)

        # Calculate bearing health index when we have enough data
        if self.sensor_state.get("crest_factor") is not None and self.sensor_state.get("velocity_rms") is not None:
            # Track baseline temperature
            current_temp = self.sensor_state.get("temperature", 25.0)
            if self.baseline_temperature is None:
                self.baseline_temperature = current_temp
            temp_delta = abs(current_temp - self.baseline_temperature)

            # IMPORTANT: When velocity is very low (stationary), CF/Kurtosis from noise are meaningless
            # Use healthy baseline values when vibration is below threshold
            MIN_VELOCITY_THRESHOLD = 0.5  # mm/s - below this, sensor is effectively stationary
            velocity_rms = self.sensor_state.get("velocity_rms", 0)

            if velocity_rms < MIN_VELOCITY_THRESHOLD:
                # Stationary: use healthy baseline values (noise produces false high CF/Kurtosis)
                effective_cf = 2.5  # Healthy baseline
                effective_kurt = 3.0  # Gaussian baseline
            else:
                # Moving: use calculated values
                effective_cf = self.sensor_state.get("crest_factor", 3.0)
                effective_kurt = self.sensor_state.get("kurtosis", 3.0)

            self.sensor_state["bearing_health_index"] = calculate_bearing_health_index(
                effective_cf,
                effective_kurt,
                velocity_rms,
                temp_delta
            )
            self.sensor_state["health_status"] = classify_health_status(
                self.sensor_state["bearing_health_index"]
            )
            self.sensor_state["temperature_delta"] = temp_delta

            # Store effective values for display (so dashboard shows meaningful values)
            self.sensor_state["crest_factor_display"] = effective_cf
            self.sensor_state["kurtosis_display"] = effective_kurt

    # =========================================================================
    # Publishing (ThingsBoard Telemetry)
    # =========================================================================

    def _build_thingsboard_telemetry(self) -> dict:
        """Build ThingsBoard telemetry payload from current sensor state."""
        # Use pre-calculated velocity from ML Feature 8 if available
        vx = self.sensor_state.get("velocity_x", 0)
        vy = self.sensor_state.get("velocity_y", 0)
        vz = self.sensor_state.get("velocity_z", 0)

        # If no pre-calculated velocity, calculate from acceleration RMS
        if vx == 0 and vy == 0 and vz == 0:
            vx = accel_to_velocity(abs(self.sensor_state["accel_x"]))
            vy = accel_to_velocity(abs(self.sensor_state["accel_y"]))
            vz = accel_to_velocity(abs(self.sensor_state["accel_z"]))

        velocity_rms = calculate_velocity_rms(vx, vy, vz)

        # Classify ISO zone
        iso_zone = classify_iso_zone(velocity_rms)

        telemetry = {
            "velocity_rms": round(velocity_rms, 4),
            "velocity_x": round(vx, 4),
            "velocity_y": round(vy, 4),
            "velocity_z": round(vz, 4),
            "iso_zone": iso_zone,
            "temperature": round(self.sensor_state["temperature"], 1),
            "pressure": round(self.sensor_state["pressure"], 1),
            "gyro_x": self.sensor_state["gyro_x"],
            "gyro_y": self.sensor_state["gyro_y"],
            "gyro_z": self.sensor_state["gyro_z"],
            "accel_x": round(self.sensor_state["accel_x"], 2),
            "accel_y": round(self.sensor_state["accel_y"], 2),
            "accel_z": round(self.sensor_state["accel_z"], 2),
        }

        # Add acceleration in g units if available
        if self.sensor_state.get("accel_rms_x_g"):
            telemetry["accel_rms_x_g"] = self.sensor_state["accel_rms_x_g"]
            telemetry["accel_rms_y_g"] = self.sensor_state["accel_rms_y_g"]
            telemetry["accel_rms_z_g"] = self.sensor_state["accel_rms_z_g"]

        # Add peak acceleration if available (ML Feature 7)
        if self.sensor_state.get("accel_peak_x"):
            telemetry["accel_peak_x"] = round(self.sensor_state["accel_peak_x"], 2)
            telemetry["accel_peak_y"] = round(self.sensor_state["accel_peak_y"], 2)
            telemetry["accel_peak_z"] = round(self.sensor_state["accel_peak_z"], 2)

        # Add AccPeak from Time Domain characteristic (in m/s²)
        if self.sensor_state.get("acc_peak_x_ms2"):
            telemetry["acc_peak_x_ms2"] = round(self.sensor_state["acc_peak_x_ms2"], 4)
            telemetry["acc_peak_y_ms2"] = round(self.sensor_state["acc_peak_y_ms2"], 4)
            telemetry["acc_peak_z_ms2"] = round(self.sensor_state["acc_peak_z_ms2"], 4)

        # Add ML status if available (ML Feature 9)
        if self.sensor_state.get("ml_status"):
            telemetry["ml_status"] = self.sensor_state["ml_status"]

        # Add FFT amplitudes if available
        if self.sensor_state.get("fft_amplitudes"):
            telemetry["fft_amplitudes"] = self.sensor_state["fft_amplitudes"]

        # Add bearing health metrics
        if self.sensor_state.get("crest_factor") is not None:
            telemetry["crest_factor"] = round(self.sensor_state["crest_factor"], 3)
            telemetry["kurtosis"] = round(self.sensor_state.get("kurtosis", 3.0), 3)
            telemetry["bearing_health_index"] = self.sensor_state.get("bearing_health_index", 100)
            telemetry["health_status"] = self.sensor_state.get("health_status", "UNKNOWN")
            telemetry["temperature_delta"] = round(self.sensor_state.get("temperature_delta", 0), 1)

        # =====================================================================
        # Anomaly Detection Integration
        # =====================================================================
        if anomaly_detector is not None:
            # Prepare features for anomaly detection
            anomaly_features = {
                "velocity_rms": velocity_rms,
                "crest_factor": telemetry.get("crest_factor", 3.0),
                "kurtosis": telemetry.get("kurtosis", 3.0),
                "temperature": telemetry.get("temperature", 25.0),
                "bearing_health_index": telemetry.get("bearing_health_index", 100)
            }

            if anomaly_detector.mode == "learn":
                # Learning mode: collect samples for baseline
                anomaly_detector.update(**anomaly_features)
                status = anomaly_detector.get_status()
                telemetry["anomaly_mode"] = "learning"
                telemetry["anomaly_progress"] = round(status["progress"], 1)
                telemetry["anomaly_samples"] = status["sample_count"]

            elif anomaly_detector.mode == "monitor":
                # Monitor mode: check for anomalies
                result = anomaly_detector.check(**anomaly_features)
                telemetry["anomaly"] = result["anomaly"]
                telemetry["anomaly_score"] = result["score"]
                telemetry["anomaly_mode"] = "monitoring"

                if result["anomaly"] and result["reasons"]:
                    # Log anomaly details
                    reasons_str = "; ".join(
                        f"{r['feature']}={r['value']:.4f} ({r['severity']}, z={r['z_score']})"
                        for r in result["reasons"]
                    )
                    logger.warning(f"ANOMALY DETECTED (score={result['score']:.2f}): {reasons_str}")
                    telemetry["anomaly_reasons"] = [r["feature"] for r in result["reasons"]]

        return telemetry

    async def poll_sensors(self):
        """Poll readable sensors directly (fallback when notifications don't work)."""
        if not self.client or not self.client.is_connected:
            return

        try:
            # Read temperature (individual characteristic uses /50.0 scaling)
            temp_uuid = "00020000-0001-11e1-ac36-0002a5d5c51b"
            data = await self.client.read_gatt_char(temp_uuid)
            if len(data) >= 9:
                temp_raw = struct.unpack_from('<H', data, 7)[0]
                temp_c = temp_raw / 50.0
                self.sensor_state["temperature"] = temp_c
                self.sensor_state["last_update"] = datetime.now(timezone.utc)

            # Read pressure (individual characteristic uses +700 offset)
            pressure_uuid = "00040000-0001-11e1-ac36-0002a5d5c51b"
            data = await self.client.read_gatt_char(pressure_uuid)
            if len(data) >= 4:
                pressure_raw = struct.unpack_from('<H', data, 2)[0]
                pressure_mbar = pressure_raw + 700.0
                self.sensor_state["pressure"] = pressure_mbar

            # Note: Gyroscope/accelerometer/velocity characteristics don't support READ
            # They only send data via NOTIFY when the device detects motion
            # For stationary sensor, velocity_rms will be 0 until vibration is detected
            # Use whatever gyro data was received via notifications
            gyro_x = abs(self.sensor_state.get("gyro_x", 0)) / 1000.0  # mdps to dps
            gyro_y = abs(self.sensor_state.get("gyro_y", 0)) / 1000.0
            gyro_z = abs(self.sensor_state.get("gyro_z", 0)) / 1000.0
            if gyro_x > 0 or gyro_y > 0 or gyro_z > 0:
                vel_x = gyro_x * 0.175
                vel_y = gyro_y * 0.175
                vel_z = gyro_z * 0.175
                vel_rms = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
                self.sensor_state["velocity_x"] = vel_x
                self.sensor_state["velocity_y"] = vel_y
                self.sensor_state["velocity_z"] = vel_z
                self.sensor_state["velocity_rms"] = vel_rms
                self.accel_buffer.append(vel_rms)

        except Exception as e:
            logger.info(f"Sensor poll error: {e}")

    async def flush_buffer(self):
        """Flush buffered readings - now just clears buffer, telemetry sent separately."""
        async with self.buffer_lock:
            # Clear buffer but don't publish individually
            # Telemetry is published as aggregated data
            self.buffer.clear()

    async def publish_telemetry(self):
        """Publish aggregated telemetry to ThingsBoard."""
        if not self.sensor_state["last_update"]:
            return  # No data yet

        telemetry = self._build_thingsboard_telemetry()
        payload = json.dumps(telemetry)
        topic = CONFIG["mqtt_topic"]

        # MQTT to ThingsBoard
        if self.mqtt_client:
            try:
                result = self.mqtt_client.publish(topic, payload, qos=1)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    self.stats["messages_sent"] += 1
                    self.stats["last_message_time"] = datetime.now(timezone.utc)
                    # Log every 5th message
                    if self.stats["messages_sent"] % 5 == 0:
                        logger.info(f"Published {self.stats['messages_sent']} telemetry messages")
                        # Show sample values
                        logger.info(f"  Temp: {telemetry.get('temperature', 0):.1f}°C, "
                                   f"Pressure: {telemetry.get('pressure', 0):.0f} mbar, "
                                   f"Vel RMS: {telemetry.get('velocity_rms', 0):.4f} mm/s")
                        # Show bearing health metrics if available
                        if telemetry.get('crest_factor') is not None:
                            logger.info(f"  Crest Factor: {telemetry.get('crest_factor', 0):.2f}, "
                                       f"Kurtosis: {telemetry.get('kurtosis', 3.0):.2f}, "
                                       f"Health Index: {telemetry.get('bearing_health_index', 100):.0f} "
                                       f"({telemetry.get('health_status', 'UNKNOWN')})")
            except Exception as e:
                logger.error(f"MQTT publish error: {e}")

        # NATS (optional)
        if self.nats_client:
            try:
                nats_subject = topic.replace("/", ".")
                await self.nats_client.publish(nats_subject, payload.encode())
                self.stats["messages_sent"] += 1
                self.stats["last_message_time"] = datetime.now(timezone.utc)
            except Exception as e:
                logger.error(f"NATS publish error: {e}")

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def monitor_connection(self):
        """Monitor connection health and reconnect if needed."""
        while self.running:
            try:
                if self.client and not self.client.is_connected:
                    logger.warning("Connection lost, attempting reconnect...")
                    self.connected = False
                    await self.connect_with_retry()
                    await self.subscribe_all()
                    await self.enable_continuous_streaming()

                await asyncio.sleep(CONFIG["health_check_interval"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
                await asyncio.sleep(1)

    async def publish_loop(self):
        """Publish aggregated telemetry to ThingsBoard at regular intervals."""
        while self.running:
            try:
                # Poll sensors every 1 second for real-time data
                await self.poll_sensors()

                # Clear buffer and publish aggregated telemetry
                await self.flush_buffer()
                await self.publish_telemetry()
                await asyncio.sleep(1.0)  # Publish telemetry every 1 second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in publish loop: {e}")
                await asyncio.sleep(1)

    async def stats_reporter(self):
        """Periodically report statistics."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Report every minute
                logger.info(
                    f"Stats: sent={self.stats['messages_sent']}, "
                    f"buffered={self.stats['messages_buffered']}, "
                    f"reconnects={self.stats['reconnect_count']}, "
                    f"connected={self.connected}"
                )
            except asyncio.CancelledError:
                break

    async def fft_acquisition_loop(self):
        """Trigger FFT acquisition every 3 seconds.

        Firmware FFT configuration:
        - TACQ_DEFAULT = 2000ms (2 seconds acquisition time)
        - FFT_SIZE_DEFAULT = 1024 samples

        We trigger every 3 seconds to allow for TACQ (2s) plus processing margin.
        """
        FFT_MASK = 0x20000000
        while self.running:
            try:
                if self.connected and self.client and self.client.is_connected:
                    await self.send_sensor_enable_command("fft", FFT_MASK)
                await asyncio.sleep(3)  # TACQ=2s + 1s margin
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"FFT acquisition error: {e}")
                await asyncio.sleep(1)

    async def run(self):
        """Main entry point."""
        logger.info("=" * 60)
        logger.info("PROTEUS BLE Gateway starting...")
        logger.info(f"Device: {CONFIG['device_name']}")
        logger.info(f"MQTT: {'enabled' if CONFIG['mqtt_enabled'] else 'disabled'}")
        logger.info(f"NATS: {'enabled' if CONFIG['nats_enabled'] else 'disabled'}")
        logger.info("=" * 60)

        # Setup brokers
        if not self.setup_mqtt():
            if CONFIG["mqtt_enabled"]:
                logger.error("MQTT setup failed, exiting")
                return

        if not await self.setup_nats():
            if CONFIG["nats_enabled"]:
                logger.error("NATS setup failed, exiting")
                return

        # Initial connection
        if not await self.connect_with_retry():
            logger.error("Failed to connect to device")
            return

        # Subscribe to characteristics
        if not await self.subscribe_all():
            logger.warning("No characteristics subscribed, but continuing...")

        # Enable continuous streaming (no need to shake the board)
        await self.enable_continuous_streaming()

        # Start background tasks
        tasks = [
            asyncio.create_task(self.monitor_connection()),
            asyncio.create_task(self.publish_loop()),
            asyncio.create_task(self.stats_reporter()),
            asyncio.create_task(self.fft_acquisition_loop()),
        ]

        logger.info("Gateway running. Press Ctrl+C to stop.")

        try:
            # Wait for shutdown signal
            while self.running:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass

        finally:
            logger.info("Shutting down...")

            # Cancel background tasks
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            # Flush remaining buffer
            await self.flush_buffer()

            # Disconnect BLE
            if self.client and self.client.is_connected:
                await self.client.disconnect()
                logger.info("BLE disconnected")

            # Stop MQTT
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                logger.info("MQTT disconnected")

            # Close NATS
            if self.nats_client:
                await self.nats_client.close()
                logger.info("NATS disconnected")

            # Save anomaly baseline if in learn mode
            if anomaly_detector is not None and anomaly_detector.mode == "learn":
                status = anomaly_detector.get_status()
                logger.info(f"Saving baseline ({status['sample_count']} samples collected)...")
                if anomaly_detector.save_baseline():
                    logger.info(f"Baseline saved to {anomaly_detector.baseline_file}")
                else:
                    logger.warning("Baseline may not be complete (need more samples)")

            logger.info("Shutdown complete")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Entry point."""
    global anomaly_detector

    import argparse
    parser = argparse.ArgumentParser(
        description="PROTEUS BLE Gateway with Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Anomaly Detection Modes:
  --mode learn     Collect baseline data from healthy machine (run 1-2 weeks)
  --mode monitor   Detect anomalies based on learned baseline
  --mode off       Disable anomaly detection (default)

Examples:
  # Step 1: Learn baseline from healthy machine
  python proteus_gateway.py --mode learn --baseline baseline.json

  # Step 2: Monitor for anomalies (after baseline is learned)
  python proteus_gateway.py --mode monitor --baseline baseline.json

  # Run with dashboard
  python proteus_gateway.py --mode monitor --dashboard
        """
    )
    parser.add_argument('--dashboard', action='store_true', help='Start local dashboard')
    parser.add_argument('--port', type=int, default=8080, help='Dashboard port (default: 8080)')
    parser.add_argument('--mode', choices=['learn', 'monitor', 'off'], default='off',
                        help='Anomaly detection mode: learn, monitor, or off (default: off)')
    parser.add_argument('--baseline', type=str, default='baseline.json',
                        help='Baseline file path (default: baseline.json)')
    args = parser.parse_args()

    # Initialize anomaly detector if enabled
    if args.mode != 'off':
        anomaly_detector = AnomalyDetector(mode=args.mode, baseline_file=args.baseline)
        logger.info(f"Anomaly detection: {args.mode.upper()} mode")
        if args.mode == 'monitor':
            status = anomaly_detector.get_status()
            if not status['baseline_ready']:
                logger.warning("No baseline loaded! Run with --mode learn first to collect baseline.")
    else:
        logger.info("Anomaly detection: DISABLED")

    # Log ThingsBoard configuration
    if CONFIG["mqtt_enabled"]:
        logger.info(f"ThingsBoard MQTT: {CONFIG['mqtt_host']}:{CONFIG['mqtt_port']}")
        logger.info(f"Device token: {CONFIG['mqtt_user'][:8]}...")

    gateway = ProteusGateway()

    if args.dashboard:
        try:
            from dashboard import ProteusDashboard, run_dashboard
            logger.info(f"Starting dashboard on port {args.port}...")
            run_dashboard(gateway, port=args.port)
        except ImportError as e:
            logger.error(f"Dashboard requires nicegui. Install with: pip install nicegui")
            logger.error(f"Import error: {e}")
            sys.exit(1)
    else:
        try:
            asyncio.run(gateway.run())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
