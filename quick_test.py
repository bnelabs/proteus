#!/usr/bin/env python3
"""
Quick PROTEUS Connection Test
Run: python3 quick_test.py
Output saved to: test_output.txt
"""

import asyncio
import sys
from bleak import BleakClient, BleakScanner
from datetime import datetime

DEVICE_ADDRESS = "2402245D-06F8-8C06-6450-9C102D4D7CE6"
OUTPUT_FILE = "test_output.txt"

# Capture output to both console and file
class TeeOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        self.file.close()

sys.stdout = TeeOutput(OUTPUT_FILE)

# ML Feature UUIDs (what we care about)
VELOCITY_RMS_UUID = "00000008-0002-11e1-ac36-0002a5d5c51b"
ACCEL_RMS_UUID = "00000006-0002-11e1-ac36-0002a5d5c51b"
FFT_UUID = "00000005-0002-11e1-ac36-0002a5d5c51b"
TEMPERATURE_UUID = "00020000-0001-11e1-ac36-0002a5d5c51b"

data_received = {"velocity": 0, "accel": 0, "fft": 0, "temp": 0}


def parse_velocity_rms(data: bytes) -> dict:
    """Parse ML Feature 8 - Velocity RMS (firmware-calculated!)"""
    if len(data) < 14:
        return None
    import struct
    ts = struct.unpack_from('<H', data, 0)[0]
    vel_x = struct.unpack_from('<f', data, 2)[0]
    vel_y = struct.unpack_from('<f', data, 6)[0]
    vel_z = struct.unpack_from('<f', data, 10)[0]
    return {
        "timestamp": ts,
        "velocity_rms_x": vel_x,
        "velocity_rms_y": vel_y,
        "velocity_rms_z": vel_z,
        "velocity_rms_total": (vel_x**2 + vel_y**2 + vel_z**2)**0.5
    }


def parse_temperature(data: bytes) -> float:
    """Parse temperature characteristic"""
    if len(data) < 4:
        return None
    import struct
    ts = struct.unpack_from('<H', data, 0)[0]
    temp_raw = struct.unpack_from('<h', data, 2)[0]
    return temp_raw / 10.0  # Temperature in °C


def notification_handler(char_uuid: str):
    def handler(sender, data):
        now = datetime.now().strftime("%H:%M:%S")
        uuid_short = char_uuid.split('-')[0]

        if char_uuid == VELOCITY_RMS_UUID:
            parsed = parse_velocity_rms(data)
            if parsed:
                data_received["velocity"] += 1
                print(f"[{now}] VELOCITY RMS: {parsed['velocity_rms_total']:.4f} mm/s "
                      f"(X={parsed['velocity_rms_x']:.3f}, Y={parsed['velocity_rms_y']:.3f}, Z={parsed['velocity_rms_z']:.3f})")

        elif char_uuid == TEMPERATURE_UUID:
            temp = parse_temperature(data)
            if temp:
                data_received["temp"] += 1
                print(f"[{now}] TEMPERATURE: {temp:.1f}°C")

        elif char_uuid == FFT_UUID:
            data_received["fft"] += 1
            print(f"[{now}] FFT: received {len(data)} bytes (packet #{data_received['fft']})")

        elif char_uuid == ACCEL_RMS_UUID:
            data_received["accel"] += 1
            if data_received["accel"] % 10 == 1:  # Print every 10th
                print(f"[{now}] ACCEL RMS: received (packet #{data_received['accel']})")

    return handler


async def main():
    print("=" * 60)
    print("PROTEUS Quick Connection Test")
    print("=" * 60)

    # Scan and connect using device object (more reliable on macOS)
    print("\n[1] Scanning for PROTEUS...")

    proteus = None
    scanner = BleakScanner()
    await scanner.start()
    await asyncio.sleep(5.0)
    await scanner.stop()

    for d in scanner.discovered_devices:
        if d.name and ("PROTEUS" in d.name.upper() or "CINGOZ" in d.name.upper()):
            proteus = d
            print(f"    Found: {d.name} ({d.address})")
            break

    if not proteus:
        print(f"    Not found by name, trying direct address: {DEVICE_ADDRESS}")
        print("    Make sure PROTEUS is powered on and not connected elsewhere.")
        return

    # Connect using the device object directly (works better on macOS)
    print("\n[2] Connecting...")
    print(f"    Using device: {proteus.name}")

    client = BleakClient(proteus, timeout=30)
    try:
        await client.connect()
        print(f"    Connected to {proteus.name}!")

        # Subscribe to key characteristics
        print("\n[3] Subscribing to sensor notifications...")

        subscribed = []
        for uuid, name in [
            (VELOCITY_RMS_UUID, "Velocity RMS"),
            (TEMPERATURE_UUID, "Temperature"),
            (FFT_UUID, "FFT Amplitude"),
            (ACCEL_RMS_UUID, "Accel RMS"),
        ]:
            try:
                await client.start_notify(uuid, notification_handler(uuid))
                subscribed.append(name)
                print(f"    ✓ {name}")
            except Exception as e:
                print(f"    ✗ {name}: {e}")

        if not subscribed:
            print("\n    ERROR: Could not subscribe to any characteristics!")
            return

        # Wait for data
        print("\n[4] Listening for 30 seconds... (Ctrl+C to stop)")
        print("-" * 60)

        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass

        print("-" * 60)
        print("\n[5] Summary:")
        print(f"    Velocity RMS packets: {data_received['velocity']}")
        print(f"    Temperature packets:  {data_received['temp']}")
        print(f"    FFT packets:          {data_received['fft']}")
        print(f"    Accel RMS packets:    {data_received['accel']}")

        if data_received['velocity'] > 0:
            print("\n    ✓ PROTEUS is working! Firmware is calculating velocity RMS.")
        else:
            print("\n    ⚠ No velocity data. May need to enable ML features.")

    finally:
        if client.is_connected:
            await client.disconnect()
            print("\n    Disconnected.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
    print(f"\n\nOutput saved to: {OUTPUT_FILE}")
    # Close file properly
    if hasattr(sys.stdout, 'file'):
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal
