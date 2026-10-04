import json
import ssl
import time

import paho.mqtt.client as mqtt

from config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_USERNAME,
    MQTT_PASSWORD,
    TELEMETRY_TOPIC,
    PUBLISH_INTERVAL,
)
from energy_simulator import generate_telemetry


def build_client():
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="nuclear_iot_simulator",
    )
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set(
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLS_CLIENT,
    )
    return client


def main():
    attack_mode = "normal"
    client = build_client()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print("Connected to MQTT broker")
        else:
            print(f"MQTT connection failed: {reason_code}")

    client.on_connect = on_connect

    print("Connecting...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    print("Nuclear IoT telemetry simulator")
    print("Commands: normal, power_spike, thermal_pressure, mqtt_flood, payload_injection, combined")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # Non-blocking simple control: read mode from a tiny local file if present.
            try:
                with open("data/attack_mode.txt", "r", encoding="utf-8") as f:
                    mode = f.read().strip()
                    if mode:
                        attack_mode = mode
            except FileNotFoundError:
                pass

            data = generate_telemetry(attack_mode=attack_mode)
            result = client.publish(TELEMETRY_TOPIC, json.dumps(data))
            result.wait_for_publish()
            print(f"[{data['timestamp']}] {attack_mode}: {data['device_id']} | {data['power']} W")
            time.sleep(PUBLISH_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping publisher...")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
