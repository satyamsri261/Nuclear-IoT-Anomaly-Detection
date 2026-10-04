"""Synthetic nuclear-facility IoT telemetry generator.

All values are simulated for a college demonstration. No real nuclear
instrumentation or control system is accessed.
"""
import random
from datetime import datetime

DEVICES = [
    "Cooling_Pump_01",
    "Cooling_Pump_02",
    "Turbine_Generator_01",
    "Aux_Power_01",
]


def generate_telemetry(device_id=None, attack_mode="normal"):
    device_id = device_id or random.choice(DEVICES)

    voltage = random.uniform(225, 238)
    current = random.uniform(3.0, 8.0)
    power_factor = random.uniform(0.90, 0.98)
    cooling_temp = random.uniform(68, 80)
    system_pressure = random.uniform(96, 104)
    vibration = random.uniform(0.10, 0.45)
    radiation_index = random.uniform(0.95, 1.05)
    message_rate = random.uniform(0.7, 1.5)
    payload_size = random.randint(180, 420)
    latency_ms = random.uniform(35, 90)
    auth_failures = 0

    # Controlled anomaly/cyberattack simulations.
    if attack_mode == "power_spike":
        voltage = random.uniform(250, 275)
        current = random.uniform(11, 16)
        power_factor = random.uniform(0.55, 0.75)

    elif attack_mode == "thermal_pressure":
        cooling_temp = random.uniform(90, 105)
        system_pressure = random.uniform(115, 130)
        vibration = random.uniform(0.9, 1.5)

    elif attack_mode == "mqtt_flood":
        message_rate = random.uniform(25, 90)
        payload_size = random.randint(700, 1800)
        latency_ms = random.uniform(180, 600)

    elif attack_mode == "payload_injection":
        payload_size = random.randint(5000, 12000)
        auth_failures = random.randint(2, 10)
        latency_ms = random.uniform(150, 500)

    elif attack_mode == "combined":
        voltage = random.uniform(250, 275)
        current = random.uniform(11, 16)
        power_factor = random.uniform(0.55, 0.75)
        cooling_temp = random.uniform(88, 105)
        message_rate = random.uniform(25, 90)
        payload_size = random.randint(5000, 12000)
        latency_ms = random.uniform(180, 600)
        auth_failures = random.randint(2, 10)

    if attack_mode in {"thermal_pressure", "combined"}:
        radiation_index = random.uniform(1.10, 1.30)
    elif attack_mode == "payload_injection":
        radiation_index = random.uniform(0.95, 1.05)

    power = voltage * current * power_factor

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device_id": device_id,
        "attack_mode": attack_mode,
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "power": round(power, 2),
        "power_factor": round(power_factor, 3),
        "energy_kwh": round(power / 1000 * 2 / 3600, 6),
        "cooling_temp": round(cooling_temp, 2),
        "system_pressure": round(system_pressure, 2),
        "vibration": round(vibration, 3),
        "radiation_index": round(radiation_index, 3),
        "message_rate": round(message_rate, 2),
        "payload_size": int(payload_size),
        "latency_ms": round(latency_ms, 2),
        "auth_failures": int(auth_failures),
    }


if __name__ == "__main__":
    print(generate_telemetry())
