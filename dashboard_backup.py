from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

PROJECT_NAME = "Nuclear IoT Intelligence"
PROJECT_SUBTITLE = "AI-Based IoT Cyberattack & Device Anomaly Detection"
PROJECT_TAG = "SIMULATION • AI/ML • MQTT"
TEAM_NAME = "Your Team Name"


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
LATEST_FILE = BASE_DIR / "data" / "latest_data.json"
HISTORY_FILE = BASE_DIR / "data" / "telemetry_history.csv"
ALERTS_FILE = BASE_DIR / "logs" / "alerts.csv"


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLE-INSPIRED DARK UI
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:radial-gradient(circle at 8% 0%,rgba(139,92,246,.25),transparent 28%),radial-gradient(circle at 92% 8%,rgba(34,211,238,.18),transparent 26%),radial-gradient(circle at 72% 92%,rgba(59,130,246,.13),transparent 30%),linear-gradient(135deg,#070914,#090d1d 50%,#070914);color:#f8fafc}
.block-container{max-width:1550px;padding-top:1.2rem;padding-bottom:2rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c1020,#080b15);border-right:1px solid rgba(139,92,246,.22)}
.hero{position:relative;overflow:hidden;border-radius:28px;padding:28px 30px;margin-bottom:18px;background:linear-gradient(120deg,rgba(139,92,246,.24),rgba(34,211,238,.10) 48%,rgba(255,255,255,.04));border:1px solid rgba(167,139,250,.30);box-shadow:0 20px 70px rgba(0,0,0,.35),inset 0 1px rgba(255,255,255,.08)}
.hero:after{content:"";position:absolute;width:280px;height:280px;right:-90px;top:-135px;border-radius:50%;background:rgba(34,211,238,.14);filter:blur(8px)}
.hero-title{font-size:35px;font-weight:800;letter-spacing:-1.5px;margin-top:9px}.hero-subtitle{color:#aab3c8;margin-top:7px;font-size:13px}
.badge{display:inline-block;color:#ede9fe;background:rgba(139,92,246,.18);border:1px solid rgba(167,139,250,.32);border-radius:999px;padding:7px 12px;font-size:10px;font-weight:800;letter-spacing:1px}
.online,.offline{padding:10px 15px;border-radius:999px;font-weight:800;font-size:11px}.online{color:#86efac;background:rgba(34,197,94,.13);border:1px solid rgba(52,211,153,.28);box-shadow:0 0 22px rgba(52,211,153,.12)}.offline{color:#fda4af;background:rgba(244,63,94,.13);border:1px solid rgba(251,113,133,.28)}
.card{position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(255,255,255,.075),rgba(255,255,255,.025));border:1px solid rgba(255,255,255,.09);border-radius:20px;padding:18px 20px;min-height:108px;box-shadow:0 14px 45px rgba(0,0,0,.22)}
.card:before{content:"";position:absolute;left:0;top:0;width:4px;height:100%;background:linear-gradient(180deg,#8b5cf6,#22d3ee);box-shadow:0 0 20px rgba(139,92,246,.65)}
.card-label{color:#929db7;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.8px}.card-value{font-size:29px;font-weight:800;letter-spacing:-1px;margin-top:8px}.card-small{color:#78839c;font-size:10px;margin-top:4px}
.section-title{font-size:21px;font-weight:800;letter-spacing:-.6px;margin:21px 0 5px}.section-subtitle{color:#7f8aa4;font-size:11px;margin-bottom:12px}
.device{background:linear-gradient(145deg,rgba(20,27,49,.90),rgba(12,17,31,.90));border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:14px 16px;margin-bottom:10px}.device-name{font-weight:700;font-size:12px}.device-meta{color:#78839c;font-size:10px;margin-top:4px}
.dot-green,.dot-red,.dot-yellow{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px}.dot-green{background:#34d399;box-shadow:0 0 13px rgba(52,211,153,.75)}.dot-yellow{background:#fbbf24;box-shadow:0 0 13px rgba(251,191,36,.75)}.dot-red{background:#fb7185;box-shadow:0 0 13px rgba(251,113,133,.75)}
.status-normal,.status-warning,.status-critical{border-radius:16px;padding:15px 18px;font-weight:650}.status-normal{color:#bbf7d0;background:linear-gradient(90deg,rgba(34,197,94,.15),rgba(34,197,94,.035));border:1px solid rgba(52,211,153,.18)}.status-warning{color:#fef3c7;background:linear-gradient(90deg,rgba(245,158,11,.16),rgba(245,158,11,.035));border:1px solid rgba(251,191,36,.18)}.status-critical{color:#ffe4e6;background:linear-gradient(90deg,rgba(244,63,94,.17),rgba(244,63,94,.035));border:1px solid rgba(251,113,133,.18)}
.info-box{background:linear-gradient(135deg,rgba(59,130,246,.14),rgba(34,211,238,.06));border:1px solid rgba(96,165,250,.20);border-radius:17px;padding:15px;color:#c7d2fe;font-size:11px;line-height:1.65}.stButton>button{border-radius:12px;background:linear-gradient(135deg,rgba(139,92,246,.18),rgba(59,130,246,.13));border:1px solid rgba(139,92,246,.30);color:#e9d5ff}.stButton>button:hover{border-color:rgba(34,211,238,.60);color:#cffafe}[data-testid="stDataFrame"]{border-radius:15px;overflow:hidden}footer{visibility:hidden}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def read_json(path: Path):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def read_csv(path: Path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def num(data, key, default=0.0):
    try:
        value = data.get(key, default)
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def clean_text(value, default="—"):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def format_time(value):
    if not value:
        return "Waiting for telemetry"
    try:
        dt = pd.to_datetime(value)
        return dt.strftime("%d %b %Y • %H:%M:%S")
    except Exception:
        return str(value)


def make_chart(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b8bbc5"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        hovermode="x unified",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color="#777b86",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,.055)",
            zeroline=False,
            color="#777b86",
        ),
    )
    return fig


def metric_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-small">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD DATA
# ============================================================
latest = read_json(LATEST_FILE)
history = read_csv(HISTORY_FILE)
alerts = read_csv(ALERTS_FILE)

if not history.empty:
    if "timestamp" in history.columns:
        history["timestamp_dt"] = pd.to_datetime(
            history["timestamp"], errors="coerce"
        )
        history = history.sort_values("timestamp_dt")

# Keep a small in-memory fallback when the CSV has not appeared yet.
if "ui_history" not in st.session_state:
    st.session_state.ui_history = []

if latest:
    snapshot = latest.copy()
    snapshot["_ui_time"] = pd.Timestamp.now()
    st.session_state.ui_history.append(snapshot)
    st.session_state.ui_history = st.session_state.ui_history[-150:]

if history.empty and st.session_state.ui_history:
    history = pd.DataFrame(st.session_state.ui_history)

if not history.empty and "timestamp_dt" not in history.columns:
    if "timestamp" in history.columns:
        history["timestamp_dt"] = pd.to_datetime(
            history["timestamp"], errors="coerce"
        )


# ============================================================
# DERIVE STATUS
# ============================================================
severity = str(
    latest.get(
        "severity",
        latest.get("alert_severity", "NORMAL")
    )
).upper()

anomaly_value = latest.get(
    "anomaly",
    latest.get(
        "is_anomaly",
        latest.get("ml_anomaly", latest.get("ml_status", "NORMAL"))
    ),
)

if isinstance(anomaly_value, str):
    anomaly_flag = anomaly_value.upper() in {
        "1", "TRUE", "YES", "ANOMALY", "ANOMALY DETECTED", "CRITICAL", "WARNING"
    }
else:
    anomaly_flag = bool(anomaly_value)

if severity not in {"NORMAL", "WARNING", "CRITICAL"}:
    severity = "CRITICAL" if anomaly_flag else "NORMAL"

system_ok = bool(latest) and severity == "NORMAL"

# Try common reason field names.
reason_value = (
    latest.get("anomaly_reasons")
    or latest.get("alert_reason")
    or latest.get("anomaly_reason")
    or latest.get("reason")
    or ""
)

if isinstance(reason_value, list):
    reasons = [str(x) for x in reason_value if str(x).strip()]
elif reason_value:
    reasons = [str(reason_value)]
else:
    reasons = []

device_id = clean_text(latest.get("device_id"), "No device")
attack_mode = clean_text(latest.get("attack_mode"), "normal").replace("_", " ").title()

# Count anomaly records if possible.
anomaly_count = 0
if not history.empty:
    for col in ["anomaly", "is_anomaly", "ml_anomaly"]:
        if col in history.columns:
            values = history[col].astype(str).str.upper()
            anomaly_count = int(
                values.isin(["1", "TRUE", "YES", "ANOMALY", "ANOMALY DETECTED"]).sum()
            )
            break

active_alerts = len(alerts) if not alerts.empty else 0


# ============================================================
# AUTO REFRESH
# ============================================================
if st_autorefresh:
    st_autorefresh(interval=3000, key="nuclear_iot_refresh")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="font-size:34px; margin-bottom:8px;">⚛️</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:21px;font-weight:800;letter-spacing:-.6px'>{PROJECT_NAME}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:#858994;font-size:11px;line-height:1.5;margin-top:7px'>{PROJECT_SUBTITLE}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "AI Anomaly Detection",
            "Cybersecurity Monitor",
            "Process & Energy",
            "Alerts & Logs",
        ],
        label_visibility="visible",
    )

    st.markdown("---")

    st.markdown(
        f"""
        <div class="info-box">
            <b>Educational prototype</b><br>
            Synthetic telemetry only. This dashboard is not connected to
            nuclear safety, protection, control, or operational systems.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption(f"Project: {TEAM_NAME}")
    st.caption("MQTT • HiveMQ Cloud • Python • Isolation Forest")


# ============================================================
# HEADER
# ============================================================
status_html = (
    '<span class="online">● SYSTEM ONLINE</span>'
    if system_ok
    else '<span class="offline">● ATTENTION REQUIRED</span>'
)

st.markdown(
    f"""
    <div class="hero">
        <div style="display:flex;justify-content:space-between;gap:20px;align-items:flex-start;">
            <div>
                <span class="badge">{PROJECT_TAG}</span>
                <div class="hero-title">⚛️ {PROJECT_NAME}</div>
                <div class="hero-subtitle">{PROJECT_SUBTITLE}</div>
            </div>
            <div>{status_html}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":

    power_kw = num(latest, "power") / 1000
    voltage = num(latest, "voltage")
    current = num(latest, "current")
    pf = num(latest, "power_factor")
    temp = num(latest, "cooling_temp")
    pressure = num(latest, "system_pressure")
    vibration = num(latest, "vibration")
    radiation = num(latest, "radiation_index")
    msg_rate = num(latest, "message_rate")
    latency = num(latest, "latency_ms")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Live Power", f"{power_kw:.2f} kW", "Current electrical load")
    with c2:
        metric_card("Voltage", f"{voltage:.1f} V", "Live telemetry")
    with c3:
        metric_card("Current", f"{current:.2f} A", "Live telemetry")
    with c4:
        metric_card("Power Factor", f"{pf:.2f}", "Electrical quality")
    with c5:
        metric_card(
            "AI Status",
            "ANOMALY" if anomaly_flag else "NORMAL",
            f"Severity: {severity.title()}",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if severity == "CRITICAL":
        st.markdown(
            "🔴 <b>Critical anomaly detected.</b> Review the AI and alert panels.",
            unsafe_allow_html=True,
        )
    elif severity == "WARNING":
        st.markdown(
            "🟡 <b>Warning condition detected.</b> Monitor the affected telemetry.",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-normal">✓ Telemetry is currently within the demonstration\'s normal operating envelope.</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.75, 1])

    with left:
        st.markdown('<div class="section-title">Live telemetry</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Real-time electrical and process behaviour</div>',
            unsafe_allow_html=True,
        )

        if not history.empty and "power" in history.columns:
            h = history.tail(100).copy()
            x = h["timestamp_dt"] if "timestamp_dt" in h.columns else h.index

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=pd.to_numeric(h["power"], errors="coerce"),
                    mode="lines",
                    name="Power",
                    line=dict(width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(124,58,237,.10)",
                )
            )
            fig.update_yaxes(title="Power (W)")
            fig.update_xaxes(title="")
            st.plotly_chart(make_chart(fig, 355), use_container_width=True)
        else:
            st.info("Waiting for telemetry history...")

    with right:
        st.markdown('<div class="section-title">Device health</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Latest device state</div>',
            unsafe_allow_html=True,
        )

        if not history.empty and "device_id" in history.columns:
            device_rows = history.copy()
            if "timestamp_dt" in device_rows.columns:
                device_rows = device_rows.sort_values("timestamp_dt")
            device_rows = device_rows.drop_duplicates(
                subset=["device_id"], keep="last"
            ).tail(8)

            for _, row in device_rows.iterrows():
                name = clean_text(row.get("device_id"), "Device")
                row_sev = str(row.get("severity", "NORMAL")).upper()
                dot = "dot-red" if row_sev == "CRITICAL" else (
                    "dot-yellow" if row_sev == "WARNING" else "dot-green"
                )
                health_text = row_sev.title()
                power = num(row, "power")
                st.markdown(
                    f"""
                    <div class="device">
                        <div class="device-name">
                            <span class="{dot}"></span>{name}
                        </div>
                        <div class="device-meta">{health_text} • {power:.0f} W</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Waiting for device telemetry...")

    st.markdown('<div class="section-title">Facility snapshot</div>', unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    with a:
        metric_card("Cooling Temperature", f"{temp:.1f} °C", "Process telemetry")
    with b:
        metric_card("System Pressure", f"{pressure:.1f}", "Synthetic units")
    with c:
        metric_card("Vibration", f"{vibration:.3f}", "Normalized signal")
    with d:
        metric_card("Radiation Index", f"{radiation:.3f}", "Synthetic normalized index")

    st.markdown("<br>", unsafe_allow_html=True)

    a, b, c = st.columns(3)
    with a:
        metric_card("MQTT Message Rate", f"{msg_rate:.2f}/s", "Communication signal")
    with b:
        metric_card("Network Latency", f"{latency:.1f} ms", "Simulated latency")
    with c:
        metric_card("Recorded Alerts", str(active_alerts), "Stored alert events")

    st.caption(f"Last telemetry: {format_time(latest.get('timestamp'))}")


# ============================================================
# AI ANOMALY PAGE
# ============================================================
elif page == "AI Anomaly Detection":

    st.markdown('<div class="section-title">🤖 AI anomaly detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Isolation Forest + transparent engineering rules</div>',
        unsafe_allow_html=True,
    )

    status_text = "ANOMALY DETECTED" if anomaly_flag else "NORMAL OPERATION"

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Model Decision", status_text, "Latest telemetry")
    with c2:
        score = latest.get("anomaly_score", latest.get("ml_score", None))
        score_text = f"{num(latest, 'anomaly_score'):.4f}" if score is not None else "N/A"
        metric_card("Anomaly Score", score_text, "Isolation Forest output")
    with c3:
        metric_card("Severity", severity, "Combined alert classification")

    st.markdown("<br>", unsafe_allow_html=True)

    if anomaly_flag:
        st.markdown(
            f'<div class="status-critical">⚠ Anomaly detected on <b>{device_id}</b>. '
            f'Current simulation mode: <b>{attack_mode}</b>.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-normal">✓ The latest telemetry is classified as normal.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Why was it flagged?</div>', unsafe_allow_html=True)

    if reasons:
        for reason in reasons:
            st.markdown(f"• {reason}")
    else:
        if anomaly_flag:
            st.info("The event was flagged, but no detailed reason was stored in the latest telemetry record.")
        else:
            st.info("No anomaly reasons are currently active.")

    st.markdown('<div class="section-title">Model input signals</div>', unsafe_allow_html=True)

    feature_names = [
        ("Voltage", "voltage", "V"),
        ("Current", "current", "A"),
        ("Power", "power", "W"),
        ("Power factor", "power_factor", ""),
        ("Cooling temperature", "cooling_temp", "°C"),
        ("System pressure", "system_pressure", ""),
        ("Vibration", "vibration", ""),
        ("Radiation index", "radiation_index", ""),
        ("MQTT message rate", "message_rate", "/s"),
        ("Payload size", "payload_size", "B"),
        ("Latency", "latency_ms", "ms"),
        ("Auth failures", "auth_failures", ""),
    ]

    cols = st.columns(4)
    for i, (label, key, unit) in enumerate(feature_names):
        with cols[i % 4]:
            value = num(latest, key)
            text_value = f"{value:.3f}" if key in {"power_factor", "vibration", "radiation_index"} else f"{value:.1f}"
            metric_card(label, f"{text_value} {unit}".strip(), "Latest feature")

    if not history.empty:
        st.markdown('<div class="section-title">Anomaly trend</div>', unsafe_allow_html=True)

        h = history.tail(120).copy()
        x = h["timestamp_dt"] if "timestamp_dt" in h.columns else h.index

        fig = go.Figure()

        if "power" in h.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=pd.to_numeric(h["power"], errors="coerce"),
                    mode="lines",
                    name="Power",
                    line=dict(width=2),
                )
            )

        if "cooling_temp" in h.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=pd.to_numeric(h["cooling_temp"], errors="coerce"),
                    mode="lines",
                    name="Cooling Temp",
                    yaxis="y2",
                    line=dict(width=2, dash="dot"),
                )
            )

        fig.update_layout(
            yaxis=dict(title="Power (W)"),
            yaxis2=dict(
                title="Temperature (°C)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )
        st.plotly_chart(make_chart(fig, 370), use_container_width=True)


# ============================================================
# CYBERSECURITY PAGE
# ============================================================
elif page == "Cybersecurity Monitor":

    st.markdown('<div class="section-title">🛡 Cybersecurity monitor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">MQTT communication and simulated security signals</div>',
        unsafe_allow_html=True,
    )

    msg_rate = num(latest, "message_rate")
    payload = num(latest, "payload_size")
    latency = num(latest, "latency_ms")
    auth = num(latest, "auth_failures")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("MQTT", "CONNECTED" if latest else "WAITING", "HiveMQ Cloud")
    with c2:
        metric_card("Message Rate", f"{msg_rate:.2f}/s", "Synthetic traffic")
    with c3:
        metric_card("Payload", f"{payload:.0f} B", "Message size")
    with c4:
        metric_card("Auth Failures", f"{auth:.0f}", "Authentication signal")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Security signals</div>', unsafe_allow_html=True)

        rows = [
            ("MQTT connection", "CONNECTED" if latest else "WAITING"),
            ("Traffic rate", "ELEVATED" if msg_rate > 5 else "NORMAL"),
            ("Payload size", "ELEVATED" if payload > 2000 else "NORMAL"),
            ("Authentication", "ATTENTION" if auth > 0 else "NORMAL"),
            ("Latency", "ELEVATED" if latency > 300 else "NORMAL"),
        ]

        for label, state in rows:
            if state in {"ATTENTION", "ELEVATED"}:
                dot = "dot-yellow"
            elif state == "CONNECTED" or state == "NORMAL":
                dot = "dot-green"
            else:
                dot = "dot-red"

            st.markdown(
                f"""
                <div class="device">
                    <div class="device-name"><span class="{dot}"></span>{label}</div>
                    <div class="device-meta">{state}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="section-title">Traffic trend</div>', unsafe_allow_html=True)

        if not history.empty and "message_rate" in history.columns:
            h = history.tail(120).copy()
            x = h["timestamp_dt"] if "timestamp_dt" in h.columns else h.index

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=pd.to_numeric(h["message_rate"], errors="coerce"),
                    mode="lines",
                    name="MQTT msg/s",
                    line=dict(width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,.10)",
                )
            )
            fig.update_yaxes(title="Messages / second")
            st.plotly_chart(make_chart(fig, 320), use_container_width=True)
        else:
            st.info("No MQTT history available yet.")

    st.markdown(
        """
        <div class="info-box">
            <b>Simulation note:</b> cybersecurity events in this prototype are
            generated through controlled synthetic telemetry. The system does not
            perform real-world network attacks or interact with nuclear control systems.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PROCESS & ENERGY PAGE
# ============================================================
elif page == "Process & Energy":

    st.markdown('<div class="section-title">⚡ Process & energy analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Electrical and process telemetry over time</div>',
        unsafe_allow_html=True,
    )

    if history.empty:
        st.info("Waiting for historical telemetry...")
    else:
        h = history.tail(150).copy()
        x = h["timestamp_dt"] if "timestamp_dt" in h.columns else h.index

        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            for key, label in [
                ("voltage", "Voltage"),
                ("current", "Current"),
            ]:
                if key in h.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=pd.to_numeric(h[key], errors="coerce"),
                            mode="lines",
                            name=label,
                        )
                    )
            fig.update_yaxes(title="Electrical value")
            st.plotly_chart(make_chart(fig, 330), use_container_width=True)

        with c2:
            fig = go.Figure()
            for key, label in [
                ("cooling_temp", "Cooling Temp"),
                ("system_pressure", "Pressure"),
            ]:
                if key in h.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=pd.to_numeric(h[key], errors="coerce"),
                            mode="lines",
                            name=label,
                        )
                    )
            fig.update_yaxes(title="Process value")
            st.plotly_chart(make_chart(fig, 330), use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            if "vibration" in h.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=pd.to_numeric(h["vibration"], errors="coerce"),
                        mode="lines",
                        name="Vibration",
                    )
                )
            st.plotly_chart(make_chart(fig, 300), use_container_width=True)

        with c2:
            fig = go.Figure()
            if "power" in h.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=pd.to_numeric(h["power"], errors="coerce"),
                        mode="lines",
                        name="Power",
                        fill="tozeroy",
                        fillcolor="rgba(124,58,237,.10)",
                    )
                )
            st.plotly_chart(make_chart(fig, 300), use_container_width=True)

        st.markdown('<div class="section-title">Recent telemetry</div>', unsafe_allow_html=True)

        display_cols = [
            c for c in [
                "timestamp",
                "device_id",
                "power",
                "voltage",
                "current",
                "cooling_temp",
                "system_pressure",
                "vibration",
                "message_rate",
                "severity",
            ]
            if c in h.columns
        ]

        if display_cols:
            st.dataframe(
                h[display_cols].tail(15),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# ALERTS PAGE
# ============================================================
elif page == "Alerts & Logs":

    st.markdown('<div class="section-title">🚨 Alerts & event logs</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Recorded anomaly and cybersecurity events</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Total Alerts", str(active_alerts), "Stored events")
    with c2:
        critical_count = 0
        if not alerts.empty and "severity" in alerts.columns:
            critical_count = int(
                alerts["severity"].astype(str).str.upper().eq("CRITICAL").sum()
            )
        metric_card("Critical", str(critical_count), "Recorded critical events")
    with c3:
        metric_card("Current State", severity, "Latest telemetry")

    st.markdown("<br>", unsafe_allow_html=True)

    if not alerts.empty:
        table = alerts.copy()

        # Put newest events first where possible.
        if "timestamp" in table.columns:
            table["_sort"] = pd.to_datetime(table["timestamp"], errors="coerce")
            table = table.sort_values("_sort", ascending=False)

        preferred = [
            c for c in [
                "timestamp",
                "device_id",
                "severity",
                "alert_type",
                "reason",
                "anomaly_reason",
                "attack_mode",
            ]
            if c in table.columns
        ]

        if preferred:
            table = table[preferred]

        st.dataframe(
            table.head(100),
            use_container_width=True,
            hide_index=True,
        )

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download alert log CSV",
            data=csv,
            file_name="nuclear_iot_alerts.csv",
            mime="text/csv",
        )
    else:
        st.info("No alert events have been recorded yet.")

    if reasons:
        st.markdown('<div class="section-title">Current event explanation</div>', unsafe_allow_html=True)
        for reason in reasons:
            st.markdown(f"• {reason}")


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div style="margin-top:35px;padding-top:18px;border-top:1px solid rgba(255,255,255,.07);
                color:#666a75;font-size:11px;text-align:center;">
        Nuclear IoT Intelligence • Educational synthetic-data prototype •
        MQTT + AI anomaly detection
    </div>
    """,
    unsafe_allow_html=True,
)