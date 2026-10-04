"""Control the publisher's synthetic attack/anomaly mode.

This only changes simulated data. It does not perform attacks against any
real system, broker, device, or nuclear infrastructure.
"""
import os

MODES = {
    "1": "normal",
    "2": "power_spike",
    "3": "thermal_pressure",
    "4": "mqtt_flood",
    "5": "payload_injection",
    "6": "combined",
}

os.makedirs("data", exist_ok=True)

print("Nuclear IoT demonstration - scenario controller")
print("1. Normal operation")
print("2. Power fluctuation")
print("3. Thermal/pressure fluctuation")
print("4. MQTT traffic flooding simulation")
print("5. Oversized payload/authentication anomaly")
print("6. Combined anomaly scenario")

choice = input("Select scenario: ").strip()
mode = MODES.get(choice)
if not mode:
    raise SystemExit("Invalid choice")

with open("data/attack_mode.txt", "w", encoding="utf-8") as f:
    f.write(mode)

print(f"Scenario set to: {mode}")
print("The running publisher will use this scenario on its next message.")
