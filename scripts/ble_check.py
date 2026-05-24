# pip install bleak asyncio
#
# This script connects directly to the ESP32 BLE GATT server.
# Do NOT use Windows Bluetooth pairing from Settings.
# BLE GATT devices like your ESP32 are usually accessed directly by code.
#
# Your ESP32:
# Device Name: ESP32_SENSOR
# Service UUID: 12345678-1234-1234-1234-1234567890ab
# Characteristic UUID: abcd1234-5678-1234-5678-abcdef123456
#
# It receives notifications in 20-byte chunks and "#" marks end of message.

import asyncio

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "ESP32_SENSOR"
CHAR_UUID = "abcd1234-5678-1234-5678-abcdef123456"

message_buffer = ""


def notification_handler(sender, data: bytearray):
    global message_buffer

    try:
        chunk = data.decode("utf-8", errors="ignore")
    except Exception:
        chunk = str(data)

    # End-of-message marker used by your ESP32 code
    if "#" in chunk:
        message_buffer += chunk.replace("#", "")

        print("\n========== FULL MESSAGE ==========")
        print(message_buffer.strip())
        print("==================================\n")

        message_buffer = ""
    else:
        message_buffer += chunk


async def find_device():
    print("Scanning for BLE devices...\n")

    devices = await BleakScanner.discover(timeout=10.0)

    for d in devices:
        print(f"Found: {d.name} | {d.address}")

        if d.name == DEVICE_NAME:
            print(f"\nTarget device found: {d.name}")
            return d

    return None


async def main():
    device = await find_device()

    if not device:
        print("\nESP32_SENSOR not found.")
        print("Check:")
        print("1. ESP32 is powered")
        print("2. BLE advertising started")
        print("3. Phone is disconnected from ESP32")
        print("4. You are close to the board")
        return

    print(f"\nConnecting to {device.name}...\n")

    async with BleakClient(device.address) as client:
        if not client.is_connected:
            print("Connection failed.")
            return

        print("Connected successfully.")
        print("Subscribing to notifications...\n")
        print("Waiting for sensor data...\n")

        await client.start_notify(CHAR_UUID, notification_handler)

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping notifications...")
            await client.stop_notify(CHAR_UUID)


if __name__ == "__main__":
    asyncio.run(main())
