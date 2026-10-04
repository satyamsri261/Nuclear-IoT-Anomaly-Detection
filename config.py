from dotenv import load_dotenv
load_dotenv()

import os

# =============================
# Nuclear-IoT demonstration config
# =============================
# IMPORTANT: this project uses synthetic telemetry for an educational demo.
# It is NOT connected to or intended to control real nuclear equipment.

MQTT_BROKER = os.getenv("MQTT_BROKER", "YOUR_CLUSTER_URL.s1.eu.hivemq.cloud")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

TELEMETRY_TOPIC = os.getenv("TELEMETRY_TOPIC", "nuclear_iot/telemetry")
ALERT_TOPIC = os.getenv("ALERT_TOPIC", "nuclear_iot/alerts")

PUBLISH_INTERVAL = float(os.getenv("PUBLISH_INTERVAL", "2"))
HISTORY_FILE = os.getenv("HISTORY_FILE", "data/telemetry_history.csv")
LATEST_FILE = os.getenv("LATEST_FILE", "data/latest_data.json")
ALERT_LOG_FILE = os.getenv("ALERT_LOG_FILE", "logs/alerts.csv")
MODEL_FILE = os.getenv("MODEL_FILE", "models/anomaly_model.joblib")

# Demo monitoring thresholds only; not nuclear safety limits.
VOLTAGE_NORMAL = (220.0, 240.0)
CURRENT_NORMAL = (2.0, 10.0)
POWER_FACTOR_NORMAL = (0.80, 0.99)
COOLING_TEMP_NORMAL = (60.0, 85.0)
SYSTEM_PRESSURE_NORMAL = (90.0, 110.0)
VIBRATION_NORMAL = (0.05, 0.80)
MAX_MESSAGE_RATE = 5.0
MAX_PAYLOAD_SIZE = 2000

# Demo electricity tariff for the dashboard.
TARIFF_PER_KWH = 8.0
