import json
import os
from datetime import datetime

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from streamlit_autorefresh import st_autorefresh

from config import (
    ALERT_LOG_FILE,
    HISTORY_FILE,
    LATEST_FILE,
    MODEL_FILE,
    TARIFF_PER_KWH,
)

st_autorefresh(interval=3000, key="live_refresh")

st.set_page_config(
    page_title="Nuclear IoT Intelligence",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.20); padding: 12px; border-radius: 12px;}
.small-note {font-size: 0.82rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)


def load_latest():
    if not os.path.exists(LATEST_FILE):
        return {}
    try:
        with open(LATEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_FILE).tail(250)
    except Exception:
        return pd.DataFrame()


def load_alerts():
    if not os.path.exists(ALERT_LOG_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(ALERT_LOG_FILE).tail(100)
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_model():
    try:
        obj = joblib.load(MODEL_FILE)
        return obj["model"], obj["features"]
    except Exception:
        return None, []


model, features = load_model()
data = load_latest()
history = load_history()
alerts = load_alerts()

# Sidebar
st.sidebar.title("⚛️ Nuclear IoT Intelligence")
st.sidebar.caption("Educational synthetic-data demonstration")
page = st.sidebar.radio("Navigation", ["Overview", "Energy Analytics", "AI Anomaly Detection", "Cybersecurity Monitor", "Alerts & Logs"])
st.sidebar.divider()
st.sidebar.info("This dashboard is a monitoring simulation. It is not connected to nuclear safety, protection, or control systems.")

# Header
h1, h2 = st.columns([4, 1])
with h1:
    st.title("⚛️ Nuclear Facility IoT Intelligence")
    st.caption("AI-Based IoT Cyberattack and Device Anomaly Detection")
with h2:
    if data:
        st.success("● SYSTEM ONLINE")
    else:
        st.warning("● WAITING FOR DATA")

if not data:
    st.warning("No telemetry has reached the dashboard yet.")
    st.code("python train_ml_model.py\npython mqtt_subscriber.py\npython mqtt_publisher.py", language="powershell")
    st.stop()

status = "ANOMALY" if bool(data.get("is_anomaly", False)) else "NORMAL"
severity = str(data.get("severity", "NORMAL"))
power_kw = float(data.get("power", 0)) / 1000
energy = float(data.get("energy_kwh", 0))
cost = energy * TARIFF_PER_KWH

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Live Power", f"{power_kw:.2f} kW")
c2.metric("Voltage", f"{float(data.get('voltage', 0)):.1f} V")
c3.metric("Current", f"{float(data.get('current', 0)):.2f} A")
c4.metric("Power Factor", f"{float(data.get('power_factor', 0)):.2f}")
c5.metric("AI Status", status)

if status == "ANOMALY":
    st.error(f"🚨 {severity}: {', '.join(data.get('reasons', []))}")
else:
    st.success("✅ Telemetry is currently within the demonstration's normal operating envelope.")

if page == "Overview":
    left, right = st.columns([2.1, 1])
    with left:
        st.subheader("📈 Real-Time Power & Facility Telemetry")
        if not history.empty and "timestamp" in history.columns:
            history["time"] = pd.to_datetime(history["timestamp"], errors="coerce")
            fig = go.Figure()
            if "power" in history:
                fig.add_trace(go.Scatter(x=history["time"], y=history["power"], mode="lines", name="Power (W)"))
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Time", yaxis_title="Power (W)")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("🏭 Device Health")
        if not history.empty and "device_id" in history.columns:
            latest_devices = history.groupby("device_id").tail(1)
            for _, row in latest_devices.iterrows():
                bad = str(row.get("severity", "NORMAL")) != "NORMAL"
                st.write(("🔴" if bad else "🟢") + f" **{row['device_id']}**")
                st.progress(0.35 if bad else 0.92)
        st.metric("Estimated Current Cost", f"₹ {cost:.2f}")

    st.subheader("📡 Current Facility Telemetry")
    a, b, c, d = st.columns(4)
    a.metric("Cooling Temperature", f"{float(data.get('cooling_temp', 0)):.1f} °C")
    b.metric("System Pressure", f"{float(data.get('system_pressure', 0)):.1f}")
    c.metric("Vibration", f"{float(data.get('vibration', 0)):.3f}")
    d.metric("MQTT Message Rate", f"{float(data.get('message_rate', 0)):.1f}/s")
    st.metric("Radiation Index", f"{float(data.get('radiation_index', 0)):.3f}")

elif page == "Energy Analytics":
    st.subheader("⚡ Energy Analytics")
    if history.empty:
        st.info("Collect more telemetry to populate analytics.")
    else:
        history["time"] = pd.to_datetime(history["timestamp"], errors="coerce")
        tabs = st.tabs(["Power", "Voltage & Current", "Power Factor", "Process Variables"])
        with tabs[0]:
            fig = go.Figure(go.Scatter(x=history["time"], y=history["power"], mode="lines+markers", name="Power"))
            fig.update_layout(height=420, yaxis_title="Power (W)")
            st.plotly_chart(fig, use_container_width=True)
        with tabs[1]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history["time"], y=history["voltage"], mode="lines", name="Voltage (V)"))
            fig.add_trace(go.Scatter(x=history["time"], y=history["current"], mode="lines", name="Current (A)", yaxis="y2"))
            fig.update_layout(height=420, yaxis_title="Voltage (V)", yaxis2=dict(title="Current (A)", overlaying="y", side="right"))
            st.plotly_chart(fig, use_container_width=True)
        with tabs[2]:
            fig = go.Figure(go.Scatter(x=history["time"], y=history["power_factor"], mode="lines", name="Power Factor"))
            fig.update_layout(height=420, yaxis_title="Power Factor")
            st.plotly_chart(fig, use_container_width=True)
        with tabs[3]:
            fig = go.Figure()
            for col in ["cooling_temp", "system_pressure", "vibration"]:
                if col in history:
                    fig.add_trace(go.Scatter(x=history["time"], y=history[col], mode="lines", name=col))
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

elif page == "AI Anomaly Detection":
    st.subheader("🤖 AI Anomaly Intelligence")
    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction", str(data.get("prediction", "N/A")))
    m2.metric("Anomaly Score", f"{float(data.get('anomaly_score', 0)):.4f}")
    m3.metric("Severity", severity)

    st.write("**Detected signals**")
    reasons = data.get("reasons", [])
    if reasons:
        for reason in reasons:
            st.warning("• " + reason)
    else:
        st.success("No abnormal signal detected.")

    st.subheader("Feature Snapshot")
    feature_values = {f: float(data.get(f, 0)) for f in features}
    st.dataframe(pd.DataFrame([feature_values]), use_container_width=True)

elif page == "Cybersecurity Monitor":
    st.subheader("🔐 IoT / MQTT Security Monitor")
    a, b, c, d = st.columns(4)
    a.metric("MQTT Traffic", f"{float(data.get('message_rate', 0)):.1f} msg/s")
    b.metric("Payload", f"{int(data.get('payload_size', 0))} bytes")
    c.metric("Latency", f"{float(data.get('latency_ms', 0)):.0f} ms")
    d.metric("Auth Failures", int(data.get('auth_failures', 0)))

    if float(data.get("message_rate", 0)) > 5:
        st.error("🚨 Possible MQTT traffic-flooding anomaly")
    elif int(data.get("payload_size", 0)) > 2000:
        st.error("🚨 Oversized MQTT payload anomaly")
    else:
        st.success("🟢 MQTT communication pattern is within the demo baseline")

    st.subheader("Security Signals")
    st.write({
        "topic": "nuclear_iot/telemetry",
        "device": data.get("device_id"),
        "message_rate": data.get("message_rate"),
        "payload_size": data.get("payload_size"),
        "latency_ms": data.get("latency_ms"),
        "authentication_failures": data.get("auth_failures"),
    })

elif page == "Alerts & Logs":
    st.subheader("🚨 Alerts & Event History")
    if alerts.empty:
        st.success("No anomaly alerts recorded.")
    else:
        st.dataframe(alerts.tail(50), use_container_width=True)
        st.download_button("Download Alert CSV", alerts.to_csv(index=False), "nuclear_iot_alerts.csv", "text/csv")

    if not history.empty:
        st.subheader("Latest Telemetry")
        st.dataframe(history.tail(25), use_container_width=True)
        st.download_button("Download Telemetry CSV", history.to_csv(index=False), "nuclear_iot_telemetry.csv", "text/csv")

st.divider()
st.caption(f"Last telemetry: {data.get('timestamp', 'N/A')} | Device: {data.get('device_id', 'N/A')} | Synthetic educational monitoring only")
