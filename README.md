# ⚛️ AI-Based IoT Cyberattack and Device Anomaly Detection

## Nuclear-industry themed educational prototype

This project adapts the mentor's Smart Industrial Energy Management reference into a **synthetic nuclear-facility IoT monitoring and anomaly-detection system**.

### What it does

- Generates synthetic telemetry for facility devices.
- Publishes telemetry through MQTT/HiveMQ Cloud.
- Receives telemetry through an MQTT subscriber.
- Runs an Isolation Forest anomaly detector.
- Applies transparent rule-based checks for large fluctuations and suspicious MQTT behavior.
- Detects simulated scenarios such as:
  - power/voltage/current fluctuations
  - thermal/pressure/vibration anomalies
  - MQTT traffic flooding patterns
  - oversized payloads/authentication anomalies
- Stores the latest telemetry, history, and alert logs.
- Presents a multi-page Streamlit dashboard inspired by industrial energy-intelligence interfaces.

> **Safety note:** all process values and attacks are synthetic. The project is for academic demonstration only. It is not connected to a real nuclear plant, reactor protection system, safety system, industrial controller, or operational network.

## Architecture

```text
Synthetic IoT Devices
        |
        v
Energy/Process Simulator
        |
        v
MQTT Publisher ---> HiveMQ Cloud ---> MQTT Subscriber
                                      |
                                      +--> Rule checks
                                      |
                                      +--> Isolation Forest
                                      |
                                      +--> latest_data.json
                                      +--> telemetry_history.csv
                                      +--> alerts.csv
                                                |
                                                v
                                      Streamlit Dashboard
```

## Setup

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install packages

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure HiveMQ

Copy `.env.example` to `.env` and enter your **new** HiveMQ credentials.

```text
MQTT_BROKER=...
MQTT_PORT=8883
MQTT_USERNAME=...
MQTT_PASSWORD=...
```

Do not commit or share `.env`.

### 4. Train the model

```powershell
python train_ml_model.py
```

This creates:

```text
models/anomaly_model.joblib
data/training_normal_data.csv
```

### 5. Start the subscriber

Terminal 1:

```powershell
python mqtt_subscriber.py
```

### 6. Start the publisher

Terminal 2:

```powershell
python mqtt_publisher.py
```

### 7. Start the dashboard

Terminal 3:

```powershell
streamlit run dashboard.py
```

## Simulating anomalies

Keep the publisher running, then open another terminal and run:

```powershell
python attack_simulator.py
```

Choose a scenario:

1. Normal operation
2. Power fluctuation
3. Thermal/pressure fluctuation
4. MQTT traffic flooding simulation
5. Oversized payload/authentication anomaly
6. Combined anomaly scenario

The scenario is written to `data/attack_mode.txt`. The running publisher reads it on its next cycle.

To return to normal:

```powershell
python reset_demo.py
```

## How the AI works

The model is an **Isolation Forest**, trained only on synthetic normal telemetry. It receives features such as:

- voltage
- current
- power
- power factor
- cooling temperature
- system pressure
- vibration
- radiation index (synthetic normalized units)
- MQTT message rate
- payload size
- latency
- authentication-failure count

The model produces a normal/anomaly prediction and an anomaly score. Transparent rules are also used so that the dashboard can explain obvious signals such as high message rate or an out-of-envelope process variable.

## Project files

```text
config.py                 Configuration
energy_simulator.py       Synthetic telemetry
mqtt_publisher.py         MQTT telemetry publisher
mqtt_subscriber.py        MQTT + AI anomaly detection
train_ml_model.py         Isolation Forest training
attack_simulator.py       Controlled synthetic scenarios
reset_demo.py             Reset to normal mode
dashboard.py              Streamlit dashboard
requirements.txt          Python dependencies
.env.example              Credential template
```
