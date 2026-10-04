import csv
import json
import os
import ssl
from datetime import datetime

import joblib
import pandas as pd
import paho.mqtt.client as mqtt

from config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_USERNAME,
    MQTT_PASSWORD,
    TELEMETRY_TOPIC,
    ALERT_TOPIC,
    LATEST_FILE,
    HISTORY_FILE,
    ALERT_LOG_FILE,
    MODEL_FILE,
)

FEATURES = [
    "voltage", "current", "power", "power_factor",
    "cooling_temp", "system_pressure", "vibration", "radiation_index",
    "message_rate", "payload_size", "latency_ms", "auth_failures",
]

os.makedirs(os.path.dirname(LATEST_FILE), exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
os.makedirs(os.path.dirname(ALERT_LOG_FILE), exist_ok=True)

try:
    MODEL_DATA = joblib.load(MODEL_FILE)
    MODEL = MODEL_DATA["model"] if isinstance(MODEL_DATA, dict) else MODEL_DATA
    MODEL_FEATURES = MODEL_DATA.get("features", FEATURES) if isinstance(MODEL_DATA, dict) else FEATURES
except Exception as exc:
    MODEL = None
    MODEL_FEATURES = FEATURES
    print("Model load warning:", exc)


def rule_flags(data):
    flags = []
    if not 220 <= float(data.get("voltage", 0)) <= 240:
        flags.append("voltage fluctuation")
    if not 2 <= float(data.get("current", 0)) <= 10:
        flags.append("current fluctuation")
    if not 0.80 <= float(data.get("power_factor", 0)) <= 0.99:
        flags.append("power-factor fluctuation")
    if float(data.get("cooling_temp", 0)) > 85:
        flags.append("thermal anomaly")
    if not 90 <= float(data.get("system_pressure", 0)) <= 110:
        flags.append("pressure anomaly")
    if float(data.get("vibration", 0)) > 0.80:
        flags.append("vibration anomaly")
    if not 0.90 <= float(data.get("radiation_index", 1.0)) <= 1.10:
        flags.append("radiation-index fluctuation (synthetic units)")
    if float(data.get("message_rate", 0)) > 5:
        flags.append("possible MQTT flooding")
    if int(data.get("payload_size", 0)) > 2000:
        flags.append("oversized MQTT payload")
    if int(data.get("auth_failures", 0)) > 0:
        flags.append("authentication failures")
    return flags


def analyse(data):
    frame = pd.DataFrame([{feature: float(data.get(feature, 0)) for feature in MODEL_FEATURES}])
    prediction = 1
    score = 0.0
    if MODEL is not None:
        prediction = int(MODEL.predict(frame)[0])
        score = float(MODEL.decision_function(frame)[0])
    flags = rule_flags(data)
    is_anomaly = prediction == -1 or bool(flags)
    if flags:
        severity = "CRITICAL" if len(flags) >= 2 or "possible MQTT flooding" in flags else "WARNING"
    else:
        severity = "NORMAL"
    if prediction == -1 and not flags:
        flags = ["ML-detected behavioral anomaly"]
        severity = "WARNING"
    return {
        "is_anomaly": is_anomaly,
        "prediction": prediction,
        "anomaly_score": score,
        "severity": severity,
        "reasons": flags,
    }


def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print("Connected. Subscribed to", TELEMETRY_TOPIC)
        client.subscribe(TELEMETRY_TOPIC)
    else:
        print("Connection failed:", reason_code)


def on_message(client, userdata, message):
    try:
        data = json.loads(message.payload.decode())
        result = analyse(data)
        enriched = {**data, **result, "analysis_timestamp": datetime.now().isoformat(timespec="seconds")}

        with open(LATEST_FILE, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=4)

        append_csv(HISTORY_FILE, {**data, "prediction": result["prediction"], "anomaly_score": result["anomaly_score"], "severity": result["severity"], "reasons": "; ".join(result["reasons"])})

        if result["is_anomaly"]:
            append_csv(ALERT_LOG_FILE, {"timestamp": data.get("timestamp"), "device_id": data.get("device_id"), "severity": result["severity"], "reasons": "; ".join(result["reasons"]), "anomaly_score": result["anomaly_score"]})
            client.publish(ALERT_TOPIC, json.dumps(enriched))

        status = "ANOMALY" if result["is_anomaly"] else "NORMAL"
        print(f"{status} | {data.get('device_id')} | {result['severity']} | {result['reasons']}")
    except Exception as exc:
        print("Message processing error:", exc)


def main():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id="nuclear_iot_anomaly_detector")
    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.on_connect = on_connect
    client.on_message = on_message
    print("Connecting to MQTT broker...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()
