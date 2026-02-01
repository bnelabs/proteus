#!/usr/bin/env python3
"""
Quick PROTEUS Connection Test - Cross-Platform (Windows, macOS, Linux)

Run: python quick_test.py
     python quick_test.py --scan        # Just scan for devices
     python quick_test.py --address XX  # Use specific address

Output saved to: test_output.txt
"""

import asyncio
import platform
import sys
import argparse
from bleak import BleakClient, BleakScanner
from datetime import datetime

# =============================================================================
# Cross-Platform Configuration
# =============================================================================

SYSTEM = platform.system()  # 'Darwin' (macOS), 'Windows', 'Linux'

# Default addresses per platform (user should update with their device)
DEFAULT_ADDRESSES = {
    "Darwin": "2402245D-06F8-8C06-6450-9C102D4D7CE6",  # macOS uses UUID
    "Windows": "XX:XX:XX:XX:XX:XX",  # Windows uses MAC address
    "Linux": "XX:XX:XX:XX:XX:XX",    # Linux uses MAC address
}

OUTPUT_FILE = "test_output.txt"

# =============================================================================
# Output Capture (to both console and file)
# =============================================================================

class TeeOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# =============================================================================
# BLE UUIDs (same across all platforms)
# =============================================================================

VELOCITY_RMS_UUID = "00000008-0002-11e1-ac36-0002a5d5c51b"
ACCEL_RMS_UUID = "00000006-0002-11e1-ac36-0002a5d5c51b"
FFT_UUID = "00000005-0002-11e1-ac36-0002a5d5c51b"
TEMPERATURE_UUID = "00020000-0001-11e1-ac36-0002a5d5c51b"

data_received = {"velocity": 0, "accel": 0, "fft": 0, "temp": 0}


# =============================================================================
# Data Parsers
# =============================================================================

def parse_velocity_rms(data: bytes) -> dict:
    """Parse ML Feature 8 - Velocity RMS (firmware-calculated)"""
    if len(data) < 20:
        return None
    import struct
    ts = struct.unpack_from('<H', data, 0)[0]
    # AccPeak at offset 2-7 (skip for now)
    # Velocity RMS floats at offset 8-19
    vel_x = struct.unpack_from('<f', data, 8)[0]
    vel_y = struct.unpack_from('<f', data, 12)[0]
    vel_z = struct.unpack_from('<f', data, 16)[0]
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
    temp_raw = struct.unpack_from('<h', data, 2)[0]
    return temp_raw / 10.0


# =============================================================================
# Notification Handler
# =============================================================================

def notification_handler(char_uuid: str):
    def handler(sender, data):
        now = datetime.now().strftime("%H:%M:%S")

        if char_uuid == VELOCITY_RMS_UUID:
            parsed = parse_velocity_rms(data)
            if parsed:
                data_received["velocity"] += 1
                print(f"[{now}] VELOCITY RMS: {parsed['velocity_rms_total']:.4f} mm/s "
                      f"(X={parsed['velocity_rms_x']:.3f}, Y={parsed['velocity_rms_y']:.3f}, "
                      f"Z={parsed['velocity_rms_z']:.3f})")

        elif char_uuid == TEMPERATURE_UUID:
            temp = parse_temperature(data)
            if temp:
                data_received["temp"] += 1
                print(f"[{now}] TEMPERATURE: {temp:.1f}°C")

        elif char_uuid == FFT_UUID:
            data_received["fft"] += 1
            if data_received["fft"] % 50 == 1:  # Print every 50th
                print(f"[{now}] FFT: received {len(data)} bytes (packet #{data_received['fft']})")

        elif char_uuid == ACCEL_RMS_UUID:
            data_received["accel"] += 1
            if data_received["accel"] % 10 == 1:
                print(f"[{now}] ACCEL RMS: received (packet #{data_received['accel']})")

    return handler


# =============================================================================
# Device Scanner
# =============================================================================

async def scan_for_proteus(timeout: float = 10.0) -> list:
    """Scan for PROTEUS devices. Returns list of (name, address) tuples."""
    print(f"Scanning for BLE devices ({timeout}s)...")

    devices = await BleakScanner.discover(timeout=timeout)

    proteus_devices = []
    for d in devices:
        name = d.name or "Unknown"
        if "PROTEUS" in name.upper() or "CINGOZ" in name.upper():
            proteus_devices.append((name, d.address))
            print(f"  ✓ Found PROTEUS: {name} [{d.address}]")

    if not proteus_devices:
        print("  No PROTEUS devices found.")
        print("\n  Other BLE devices nearby:")
        for d in sorted(devices, key=lambda x: x.name or "ZZZ")[:10]:
            if d.name:
                print(f"    - {d.name} [{d.address}]")

    return proteus_devices


# =============================================================================
# Main Test Function
# =============================================================================

async def main(device_address: str = None, scan_only: bool = False):
    print("=" * 60)
    print("PROTEUS Quick Connection Test")
    print("=" * 60)
    print(f"Platform: {SYSTEM} ({platform.platform()})")
    print(f"Python:   {platform.python_version()}")

    # Platform-specific notes
    if SYSTEM == "Windows":
        print("\nNote: Windows uses MAC address format (XX:XX:XX:XX:XX:XX)")
    elif SYSTEM == "Darwin":
        print("\nNote: macOS uses UUID format (XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX)")
    else:
        print("\nNote: Linux uses MAC address format (XX:XX:XX:XX:XX:XX)")

    # Step 1: Scan for devices
    print("\n" + "-" * 60)
    print("[1] Scanning for PROTEUS...")
    print("-" * 60)

    proteus_devices = await scan_for_proteus(timeout=8.0)

    if scan_only:
        print("\nScan complete.")
        return

    # Determine which address to use
    if device_address:
        address = device_address
        print(f"\nUsing provided address: {address}")
    elif proteus_devices:
        address = proteus_devices[0][1]
        print(f"\nUsing discovered device: {proteus_devices[0][0]} [{address}]")
    else:
        address = DEFAULT_ADDRESSES.get(SYSTEM, "")
        if "XX:XX" in address:
            print("\n⚠ No PROTEUS found and no valid default address.")
            print("  Run with --scan to find your device, then use --address")
            print(f"  Example: python quick_test.py --address AA:BB:CC:DD:EE:FF")
            return
        print(f"\nUsing default address: {address}")

    # Step 2: Connect
    print("\n" + "-" * 60)
    print("[2] Connecting...")
    print("-" * 60)

    client = BleakClient(address, timeout=30)

    try:
        await client.connect()
        print(f"    ✓ Connected to {address}")

        # Step 3: Subscribe to notifications
        print("\n" + "-" * 60)
        print("[3] Subscribing to sensor notifications...")
        print("-" * 60)

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

        # Step 4: Listen for data
        print("\n" + "-" * 60)
        print("[4] Listening for 30 seconds... (Ctrl+C to stop)")
        print("-" * 60)

        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            pass

        # Step 5: Summary
        print("\n" + "-" * 60)
        print("[5] Summary")
        print("-" * 60)
        print(f"    Velocity RMS packets: {data_received['velocity']}")
        print(f"    Temperature packets:  {data_received['temp']}")
        print(f"    FFT packets:          {data_received['fft']}")
        print(f"    Accel RMS packets:    {data_received['accel']}")

        if data_received['velocity'] > 0:
            print("\n    ✓ PROTEUS is working! Firmware is calculating velocity RMS.")
        else:
            print("\n    ⚠ No velocity data received.")
            print("      Try tapping/shaking the PROTEUS board to trigger vibration.")

    except Exception as e:
        print(f"    ✗ Connection failed: {e}")
        if SYSTEM == "Windows" and "XX:XX" in address:
            print("\n    Hint: You need to find your PROTEUS MAC address.")
            print("    Run: python quick_test.py --scan")

    finally:
        if client.is_connected:
            await client.disconnect()
            print("\n    Disconnected.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PROTEUS BLE Connection Test (Cross-Platform)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_test.py                    # Auto-scan and connect
  python quick_test.py --scan             # Just scan for devices
  python quick_test.py --address XX:XX:XX:XX:XX:XX  # Use specific address

Platform Notes:
  macOS:   Uses UUID format (auto-detected)
  Windows: Uses MAC address format (AA:BB:CC:DD:EE:FF)
  Linux:   Uses MAC address format (AA:BB:CC:DD:EE:FF)
        """
    )
    parser.add_argument('--scan', action='store_true',
                        help='Only scan for devices, don\'t connect')
    parser.add_argument('--address', type=str, default=None,
                        help='Device address (MAC or UUID depending on platform)')
    args = parser.parse_args()

    # Setup output capture
    sys.stdout = TeeOutput(OUTPUT_FILE)

    try:
        asyncio.run(main(device_address=args.address, scan_only=args.scan))
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\n\nError: {e}")

    print(f"\n\nOutput saved to: {OUTPUT_FILE}")

    # Restore stdout
    if hasattr(sys.stdout, 'file'):
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal
