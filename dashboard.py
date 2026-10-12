"""
Nuclear IoT Intelligence - AI-Based IoT Cyberattack & Device Anomaly Detection
Professional dark dashboard: simulated MQTT telemetry + SWaT research dataset.
Drop-in replacement for your dashboard.py (same files, same model format).

Expected files (relative to this file):
    data/latest_data.json                    data/telemetry_history.csv      logs/alerts.csv
    data/swat/dashboard/normal_sample.csv    data/swat/dashboard/attack_sample.csv
    models/swat_anomaly_model.joblib  ({"model":..., "features":[...]})      models/swat_scaler.joblib
Optional (smooth auto-refresh):  pip install streamlit-autorefresh
"""
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

PROJECT_NAME = "Nuclear IoT Intelligence"
PROJECT_SUBTITLE = "AI-Based IoT Cyberattack & Device Anomaly Detection"
PROJECT_TAG = "AI / ML  •  IoT  •  MQTT  •  CYBERSECURITY"
TEAM_NAME = "Your Team Name"

# ============================================================
# PATHS
# ============================================================
BASE = Path(__file__).resolve().parent
LIVE = BASE / "data" / "latest_data.json"
HIST = BASE / "data" / "telemetry_history.csv"
ALERTS = BASE / "logs" / "alerts.csv"
SWAT_N = BASE / "data" / "swat" / "dashboard" / "normal_sample.csv"
SWAT_A = BASE / "data" / "swat" / "dashboard" / "attack_sample.csv"
MODEL = BASE / "models" / "swat_anomaly_model.joblib"
SCALER = BASE / "models" / "swat_scaler.joblib"
LABEL = "Normal/Attack"

# ============================================================
# DESIGN TOKENS
# ============================================================
C = dict(cyan="#22d3ee", violet="#a78bfa", blue="#60a5fa", green="#34d399",
         amber="#fbbf24", red="#fb7185", pink="#f472b6")
COLORWAY = [C["cyan"], C["violet"], C["amber"], C["green"], C["pink"], C["blue"]]
SEV_COLOR = {"NORMAL": C["green"], "WARNING": C["amber"], "CRITICAL": C["red"]}
TRUE_SET = {"1", "TRUE", "YES", "ANOMALY", "ANOMALY DETECTED", "CRITICAL", "WARNING"}

st.set_page_config(page_title=PROJECT_NAME, page_icon="☢️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html,body,[class*="css"]{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:
  radial-gradient(1100px 520px at 6% -8%,rgba(139,92,246,.26),transparent 60%),
  radial-gradient(900px 480px at 98% 0%,rgba(34,211,238,.18),transparent 60%),
  radial-gradient(800px 600px at 70% 105%,rgba(59,130,246,.14),transparent 60%),#060913;color:#f1f5f9}
.block-container{max-width:1560px;padding-top:1.1rem;padding-bottom:2rem}
header[data-testid="stHeader"]{background:transparent}footer{visibility:hidden}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-thumb{background:rgba(148,163,184,.25);border-radius:8px}

/* sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1021,#070a15);border-right:1px solid rgba(139,92,246,.20)}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:6px}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);
  border-radius:12px;padding:9px 12px;width:100%;transition:all .18s ease}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover{background:rgba(139,92,246,.14);border-color:rgba(167,139,250,.38);transform:translateX(2px)}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked){background:linear-gradient(90deg,rgba(139,92,246,.30),rgba(34,211,238,.10));border-color:rgba(167,139,250,.55)}
[data-testid="stSidebar"] .stRadio label>div:first-child{display:none}
.side-brand{display:flex;align-items:center;gap:12px;margin-bottom:4px}
.side-logo{width:44px;height:44px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:23px;background:linear-gradient(135deg,#8b5cf6,#22d3ee);box-shadow:0 8px 26px rgba(139,92,246,.45)}
.side-title{font-size:17px;font-weight:800;letter-spacing:-.5px;line-height:1.1}
.side-sub{color:#8b97b1;font-size:10.5px;line-height:1.5;margin:8px 0 14px}
.nav-label{color:#64708a;font-size:10px;font-weight:700;letter-spacing:1.4px;margin:6px 0 8px}
.side-stat{display:flex;justify-content:space-between;font-size:11px;color:#8b97b1;padding:6px 0;border-bottom:1px dashed rgba(255,255,255,.07)}
.side-stat b{color:#e2e8f0;font-family:'JetBrains Mono',monospace;font-weight:500}

/* hero */
.hero{position:relative;overflow:hidden;border-radius:26px;padding:26px 30px;margin-bottom:18px;
  background:linear-gradient(120deg,rgba(139,92,246,.26),rgba(34,211,238,.10) 52%,rgba(255,255,255,.03));
  border:1px solid rgba(167,139,250,.30);box-shadow:0 24px 70px rgba(0,0,0,.40),inset 0 1px rgba(255,255,255,.09)}
.hero:before{content:"";position:absolute;inset:0;opacity:.35;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.05) 1px,transparent 1px);
  background-size:34px 34px;mask-image:linear-gradient(90deg,transparent,#000 40%,transparent)}
.hero:after{content:"";position:absolute;width:300px;height:300px;right:-100px;top:-150px;border-radius:50%;background:rgba(34,211,238,.16);filter:blur(10px)}
.hero-row{position:relative;z-index:2;display:flex;justify-content:space-between;gap:20px;align-items:flex-start;flex-wrap:wrap}
.hero-title{font-size:36px;font-weight:800;letter-spacing:-1.6px;margin-top:10px;background:linear-gradient(90deg,#fff,#c4b5fd 55%,#67e8f9);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-subtitle{color:#aab3c8;margin-top:6px;font-size:13px}
.badge{display:inline-block;color:#ede9fe;background:rgba(139,92,246,.20);border:1px solid rgba(167,139,250,.35);border-radius:999px;padding:6px 13px;font-size:10px;font-weight:800;letter-spacing:1.3px}
.pill{display:inline-flex;align-items:center;gap:8px;padding:9px 15px;border-radius:999px;font-weight:800;font-size:11px;letter-spacing:.6px;margin-left:8px;margin-bottom:6px}
.pill-ok{color:#86efac;background:rgba(34,197,94,.13);border:1px solid rgba(52,211,153,.30)}
.pill-warn{color:#fde68a;background:rgba(245,158,11,.13);border:1px solid rgba(251,191,36,.30)}
.pill-bad{color:#fda4af;background:rgba(244,63,94,.14);border:1px solid rgba(251,113,133,.34)}
.pill-idle{color:#cbd5e1;background:rgba(148,163,184,.12);border:1px solid rgba(148,163,184,.28)}
.pulse{width:9px;height:9px;border-radius:50%;background:currentColor;animation:pulse 1.6s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 currentColor}70%{box-shadow:0 0 0 9px transparent}100%{box-shadow:0 0 0 0 transparent}}

/* cards */
.card{position:relative;overflow:hidden;border-radius:20px;padding:17px 19px 15px;min-height:112px;background:linear-gradient(150deg,rgba(255,255,255,.08),rgba(255,255,255,.02));
  border:1px solid rgba(255,255,255,.09);box-shadow:0 14px 42px rgba(0,0,0,.26);transition:transform .2s,border-color .2s;margin-bottom:12px}
.card:hover{transform:translateY(-3px);border-color:rgba(167,139,250,.40)}
.card:before{content:"";position:absolute;left:0;top:0;width:4px;height:100%;background:linear-gradient(180deg,#8b5cf6,#22d3ee);box-shadow:0 0 18px rgba(139,92,246,.7)}
.card.ok:before{background:linear-gradient(180deg,#34d399,#22d3ee);box-shadow:0 0 18px rgba(52,211,153,.7)}
.card.warn:before{background:linear-gradient(180deg,#fbbf24,#fb923c);box-shadow:0 0 18px rgba(251,191,36,.7)}
.card.bad:before{background:linear-gradient(180deg,#fb7185,#f43f5e);box-shadow:0 0 18px rgba(251,113,133,.8)}
.card-top{display:flex;justify-content:space-between;align-items:center}
.card-label{color:#93a0ba;font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.9px}
.card-icon{font-size:15px;opacity:.9}
.card-value{font-size:27px;font-weight:800;letter-spacing:-1px;margin-top:9px;font-variant-numeric:tabular-nums}
.card-small{color:#74809a;font-size:10.5px;margin-top:5px}

/* sections */
.section-title{font-size:20px;font-weight:800;letter-spacing:-.5px;margin:22px 0 3px;display:flex;align-items:center;gap:10px}
.section-title:before{content:"";width:5px;height:20px;border-radius:4px;background:linear-gradient(180deg,#8b5cf6,#22d3ee)}
.section-subtitle{color:#7d89a3;font-size:11.5px;margin:0 0 12px 15px}
.panel{background:linear-gradient(145deg,rgba(255,255,255,.05),rgba(255,255,255,.015));border:1px solid rgba(255,255,255,.07);border-radius:20px;padding:6px 10px 2px}
.banner{border-radius:16px;padding:15px 19px;font-weight:600;font-size:13.5px;display:flex;gap:12px;align-items:center;margin:4px 0 8px}
.banner-ok{color:#bbf7d0;background:linear-gradient(90deg,rgba(34,197,94,.16),rgba(34,197,94,.03));border:1px solid rgba(52,211,153,.22)}
.banner-warn{color:#fef3c7;background:linear-gradient(90deg,rgba(245,158,11,.18),rgba(245,158,11,.03));border:1px solid rgba(251,191,36,.24)}
.banner-bad{color:#ffe4e6;background:linear-gradient(90deg,rgba(244,63,94,.22),rgba(244,63,94,.03));border:1px solid rgba(251,113,133,.30);animation:glow 2s infinite}
.banner-info{color:#bfdbfe;background:linear-gradient(90deg,rgba(59,130,246,.18),rgba(34,211,238,.04));border:1px solid rgba(96,165,250,.24)}
@keyframes glow{50%{box-shadow:0 0 28px rgba(244,63,94,.28)}}
.info-box{background:linear-gradient(135deg,rgba(59,130,246,.14),rgba(34,211,238,.06));border:1px solid rgba(96,165,250,.22);border-radius:16px;padding:14px 16px;color:#c7d2fe;font-size:11.5px;line-height:1.65}
.device{display:block;background:linear-gradient(145deg,rgba(20,27,49,.92),rgba(12,17,31,.92));border:1px solid rgba(255,255,255,.07);border-radius:15px;padding:12px 15px;margin-bottom:9px}
.device-name{font-weight:700;font-size:12.5px;display:flex;align-items:center;justify-content:space-between}
.device-meta{color:#7d89a3;font-size:10.5px;margin-top:3px}
.chip{font-size:9.5px;font-weight:800;letter-spacing:.8px;padding:4px 10px;border-radius:999px}
.chip-ok{color:#86efac;background:rgba(34,197,94,.13)}.chip-warn{color:#fde68a;background:rgba(245,158,11,.14)}.chip-bad{color:#fda4af;background:rgba(244,63,94,.16)}
.bar-wrap{height:7px;background:rgba(255,255,255,.07);border-radius:8px;overflow:hidden;margin-top:9px}.bar-fill{height:100%;border-radius:8px}
.flow{display:flex;gap:10px;align-items:stretch;flex-wrap:wrap}
.flow-step{flex:1;min-width:150px;border-radius:16px;padding:14px 15px;background:linear-gradient(145deg,rgba(255,255,255,.06),rgba(255,255,255,.015));border:1px solid rgba(255,255,255,.08)}
.flow-step b{display:block;font-size:12.5px;margin:6px 0 3px}.flow-step span{color:#7d89a3;font-size:10.5px;line-height:1.5}
.flow-num{display:inline-flex;width:24px;height:24px;border-radius:8px;align-items:center;justify-content:center;font-size:11px;font-weight:800;background:linear-gradient(135deg,#8b5cf6,#22d3ee);color:#fff}
.flow-arrow{align-self:center;color:#475569;font-size:18px}
.stButton>button,.stDownloadButton>button{border-radius:12px;background:linear-gradient(135deg,rgba(139,92,246,.25),rgba(59,130,246,.18));border:1px solid rgba(139,92,246,.38);color:#ede9fe;font-weight:600}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:rgba(34,211,238,.7);color:#cffafe}
[data-testid="stDataFrame"]{border-radius:15px;overflow:hidden;border:1px solid rgba(255,255,255,.07)}
[data-testid="stExpander"]{border:1px solid rgba(255,255,255,.08);border-radius:14px;background:rgba(255,255,255,.02)}
.footer{margin-top:36px;padding-top:16px;border-top:1px solid rgba(255,255,255,.07);color:#5d6880;font-size:11px;text-align:center;line-height:1.7}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# GENERIC HELPERS
# ============================================================
def read_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_csv(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def num(d, key, default=0.0):
    try:
        v = d.get(key, default)
        return default if v is None else float(v)
    except Exception:
        return default


def hex_to_rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def plot(fig, **kw):
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kw)


def show_table(data, **kw):
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=True, **kw)


def make_chart(fig, height=330):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=18, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", colorway=COLORWAY, font=dict(color="#aab3c8", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0f172a", bordercolor="#334155", font=dict(color="#f1f5f9")),
        xaxis=dict(showgrid=False, zeroline=False, color="#64708a"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.05)", zeroline=False, color="#64708a"))
    return fig


def metric_card(label, value, sub="", icon="", tone=""):
    st.markdown(
        f"""<div class="card {tone}"><div class="card-top"><div class="card-label">{label}</div>
        <div class="card-icon">{icon}</div></div><div class="card-value">{value}</div>
        <div class="card-small">{sub}</div></div>""", unsafe_allow_html=True)


def section(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def banner(kind, text):
    icon = {"ok": "✅", "warn": "⚠️", "bad": "🚨", "info": "ℹ️"}[kind]
    st.markdown(f'<div class="banner banner-{kind}"><span style="font-size:20px">{icon}</span><span>{text}</span></div>',
                unsafe_allow_html=True)


def gap(px=10):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


def signal_row(label, state, value_text, pct, tone):
    color = {"ok": C["green"], "warn": C["amber"], "bad": C["red"]}[tone]
    chip = {"ok": "chip-ok", "warn": "chip-warn", "bad": "chip-bad"}[tone]
    st.markdown(
        f"""<div class="device"><div class="device-name"><span>{label}</span><span class="chip {chip}">{state}</span></div>
        <div class="device-meta">{value_text}</div>
        <div class="bar-wrap"><div class="bar-fill" style="width:{max(3, min(100, pct)):.0f}%;background:{color}"></div></div></div>""",
        unsafe_allow_html=True)


def donut(labels, values, colors, height=300, center=""):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.70, sort=False, textinfo="percent",
                           marker=dict(colors=colors, line=dict(color="#060913", width=3))))
    fig.update_layout(showlegend=True, annotations=[dict(text=center, x=.5, y=.5, showarrow=False,
                      font=dict(size=20, color="#f1f5f9"))])
    fig = make_chart(fig, height)
    fig.update_layout(hovermode="closest", legend=dict(orientation="h", y=-0.05, x=.2))
    return fig


def line_trace(x, y, name, color, fill=False, dash=None):
    kw = dict(x=x, y=pd.to_numeric(y, errors="coerce"), mode="lines", name=name,
              line=dict(width=2.3, color=color, shape="spline", smoothing=.5, dash=dash))
    if fill:
        kw.update(fill="tozeroy", fillcolor=hex_to_rgba(color, .10))
    return go.Scatter(**kw)


# ============================================================
# SWaT MODEL + DATA (cached)
# ============================================================
@st.cache_resource
def swat_model():
    p = joblib.load(MODEL)
    return p["model"], list(p["features"]), joblib.load(SCALER)


@st.cache_data
def swat_data():
    n, a = pd.read_csv(SWAT_N), pd.read_csv(SWAT_A)
    n.columns, a.columns = n.columns.str.strip(), a.columns.str.strip()
    for d in (n, a):
        if LABEL in d:
            d[LABEL] = (d[LABEL].astype(str).str.strip().str.replace(" ", "", regex=False).str.lower()
                        .replace({"normal": "Normal", "attack": "Attack"}))
    return n, a


@st.cache_data(show_spinner="Running Isolation Forest on SWaT data...")
def run_detection(kind, size, seed=42):
    """Return (scored dataframe, top-deviating-sensors dataframe)."""
    model, features, scaler = swat_model()
    normal, attack = swat_data()
    data = attack if kind == "attack" else normal
    if size < len(data):
        data = data.sample(size, random_state=seed)
    data = data.copy()
    if "Timestamp" in data:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"].astype(str).str.strip(), errors="coerce", dayfirst=True)
        data = data.sort_values("Timestamp")
    X = data.reindex(columns=features).replace([np.inf, -np.inf], np.nan).fillna(0)
    Z = scaler.transform(X)
    data["AI Prediction"] = np.where(model.predict(Z) == -1, "ANOMALY", "NORMAL")
    data["Anomaly Score"] = model.decision_function(Z)
    bad = (data["AI Prediction"] == "ANOMALY").values
    dev = pd.DataFrame()
    if bad.any() and (~bad).any():
        a_z, n_z = np.abs(Z[bad]).mean(axis=0), np.abs(Z[~bad]).mean(axis=0)
        dev = pd.DataFrame({"Sensor": features, "Anomalous": a_z, "Normal": n_z, "Lift": a_z - n_z})
        dev = dev.sort_values("Lift", ascending=False).head(10)
    return data, dev


def performance(data):
    """Attack = positive class. Returns dict or None when labels are missing."""
    if LABEL not in data:
        return None
    y, p = (data[LABEL] == "Attack").values, (data["AI Prediction"] == "ANOMALY").values
    tp, fp, fn, tn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum()), int((~y & ~p).sum())
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=div(2 * prec * rec, prec + rec),
                accuracy=div(tp + tn, len(y)), fpr=div(fp, fp + tn))


def confusion_fig(m):
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Pred NORMAL", "Pred ANOMALY"], y=["True Normal", "True Attack"],
                               text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}",
                               textfont=dict(size=18, color="#fff"), showscale=False, hoverinfo="skip",
                               colorscale=[[0, "#111a33"], [.5, "#4c1d95"], [1, "#0891b2"]]))
    fig = make_chart(fig, 300)
    fig.update_layout(hovermode="closest", yaxis=dict(autorange="reversed", color="#aab3c8"),
                      xaxis=dict(color="#aab3c8"))
    return fig


def perf_cards(m):
    c = st.columns(4)
    with c[0]:
        metric_card("Precision", f"{m['precision']:.1%}", "Flagged rows that were real attacks", "🎯")
    with c[1]:
        metric_card("Recall", f"{m['recall']:.1%}", "Real attacks the AI caught", "📡")
    with c[2]:
        metric_card("F1 Score", f"{m['f1']:.1%}", "Balance of precision & recall", "⚖️")
    with c[3]:
        metric_card("False-alarm rate", f"{m['fpr']:.1%}", "Normal rows wrongly flagged", "🔔",
                    "ok" if m["fpr"] < .05 else "warn")


def style_predictions(df):
    def col(v):
        return {"ANOMALY": f"color:{C['red']};font-weight:700", "NORMAL": f"color:{C['green']}",
                "Attack": f"color:{C['red']}", "Normal": f"color:{C['green']}"}.get(v, "")
    cols = [c for c in ("AI Prediction", LABEL) if c in df.columns]
    s = df.style
    return (s.map if hasattr(s, "map") else s.applymap)(col, subset=cols) if cols else s


# ============================================================
# LIVE MQTT DATA
# ============================================================
latest = read_json(LIVE)
hist = read_csv(HIST)
alerts = read_csv(ALERTS)
if not hist.empty:
    ts_col = "timestamp" if "timestamp" in hist.columns else None
    if ts_col:
        hist["_t"] = pd.to_datetime(hist[ts_col], errors="coerce")
        hist = hist.sort_values("_t")
    flag = pd.Series(False, index=hist.index)
    for col in ("anomaly", "is_anomaly", "ml_anomaly"):
        if col in hist.columns:
            flag |= hist[col].astype(str).str.upper().isin(TRUE_SET)
    if "severity" in hist.columns:
        flag |= hist["severity"].astype(str).str.upper().isin(["WARNING", "CRITICAL"])
    hist["_flag"] = flag

latest = latest or {}
severity = str(latest.get("severity", latest.get("alert_severity", "NORMAL"))).upper()
av = latest.get("anomaly", latest.get("is_anomaly", latest.get("ml_anomaly", "NORMAL")))
anomaly_flag = (av.upper() in TRUE_SET) if isinstance(av, str) else bool(av)
if severity not in SEV_COLOR:
    severity = "CRITICAL" if anomaly_flag else "NORMAL"
sev_tone = {"NORMAL": "ok", "WARNING": "warn", "CRITICAL": "bad"}[severity]

swat_files_ok = SWAT_N.exists() and SWAT_A.exists()
swat_model_ok = MODEL.exists() and SCALER.exists()
swat_ready = swat_files_ok and swat_model_ok

# ============================================================
# SIDEBAR
# ============================================================
NAV = {
    "🏠  Overview": "Overview",
    "🟢  Live MQTT Demo": "Live MQTT Demo",
    "🔬  SWaT Research Mode": "SWaT Research Mode",
    "🤖  AI Anomaly Detection": "AI Anomaly Detection",
    "🛡️  Cybersecurity Monitor": "Cybersecurity Monitor",
    "🚨  Alerts & Logs": "Alerts & Logs",
}
LIVE_PAGES = {"Overview", "Live MQTT Demo", "Cybersecurity Monitor", "Alerts & Logs"}
tick = lambda ok: "✔ ready" if ok else "✖ missing"

with st.sidebar:
    st.markdown(
        f"""<div class="side-brand"><div class="side-logo">☢️</div><div class="side-title">{PROJECT_NAME}</div></div>
        <div class="side-sub">{PROJECT_SUBTITLE}</div><div class="nav-label">DASHBOARD MODE</div>""",
        unsafe_allow_html=True)
    mode = NAV[st.radio("Dashboard Mode", list(NAV), label_visibility="collapsed")]
    st.markdown('<div class="nav-label" style="margin-top:18px">DATA SOURCES</div>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="side-stat"><span>MQTT telemetry</span><b>{'● live' if latest else '○ waiting'}</b></div>
        <div class="side-stat"><span>SWaT datasets</span><b>{tick(swat_files_ok)}</b></div>
        <div class="side-stat"><span>SWaT AI model</span><b>{tick(swat_model_ok)}</b></div>
        <div class="side-stat"><span>Stored alerts</span><b>{len(alerts):,}</b></div>
        <div class="side-stat"><span>Local time</span><b>{datetime.now().strftime('%H:%M:%S')}</b></div>""",
        unsafe_allow_html=True)
    auto = st.toggle("Auto-refresh live pages", value=True)
    refresh_s = st.slider("Refresh every (s)", 2, 15, 3, disabled=not auto)
    st.markdown(
        """<div class="info-box" style="margin-top:12px"><b>🟢 MQTT</b> = safe simulated live telemetry<br>
        <b>🔬 SWaT</b> = real industrial-control research dataset<br><br>
        Educational prototype. Not connected to any real nuclear facility.</div>""",
        unsafe_allow_html=True)
    st.caption(f"Project: {TEAM_NAME}")

if auto and st_autorefresh and mode in LIVE_PAGES:
    st_autorefresh(interval=refresh_s * 1000, key="nuclear_refresh")

# ============================================================
# HERO
# ============================================================
mq = ('<span class="pill pill-ok"><span class="pulse"></span>MQTT LIVE</span>' if latest
      else '<span class="pill pill-idle"><span class="pulse"></span>MQTT WAITING</span>')
sw = ('<span class="pill pill-ok"><span class="pulse"></span>SWaT AI READY</span>' if swat_ready
      else '<span class="pill pill-warn"><span class="pulse"></span>SWaT NOT READY</span>')
al = (f'<span class="pill pill-bad"><span class="pulse"></span>ALERT • {severity}</span>'
      if latest and severity != "NORMAL" else "")
st.markdown(
    f"""<div class="hero"><div class="hero-row"><div><span class="badge">{PROJECT_TAG}</span>
    <div class="hero-title">☢️ {PROJECT_NAME}</div><div class="hero-subtitle">{PROJECT_SUBTITLE}</div></div>
    <div>{mq}{sw}{al}</div></div></div>""", unsafe_allow_html=True)

SWAT_NOTE = ("SWaT (Secure Water Treatment testbed) is a real industrial-control-system dataset used here as a "
             "research stand-in. It is <b>not</b> nuclear-plant telemetry. MQTT mode is simulated.")


def swat_unavailable():
    banner("warn", "SWaT files or model not found. Check <code>data/swat/dashboard/</code> and "
                   "<code>models/swat_anomaly_model.joblib</code> + <code>swat_scaler.joblib</code>.")


# ============================================================
# OVERVIEW
# ============================================================
if mode == "Overview":
    normal = attack = pd.DataFrame()
    n_feat = "—"
    if swat_ready:
        try:
            normal, attack = swat_data()
            n_feat = str(len(swat_model()[1]))
        except Exception as e:
            banner("warn", f"Could not load SWaT assets: {e}")

    c = st.columns(4)
    with c[0]:
        metric_card("MQTT Demo", "ONLINE" if latest else "WAITING", "Simulated live telemetry", "🟢",
                    "ok" if latest else "warn")
    with c[1]:
        metric_card("SWaT Research", "READY" if not normal.empty else "NOT READY", "Real ICS dataset + AI model", "🔬",
                    "ok" if not normal.empty else "warn")
    with c[2]:
        metric_card("SWaT Records", f"{len(normal) + len(attack):,}", "Normal + attack samples", "🗃️")
    with c[3]:
        metric_card("SWaT Features", n_feat, "Sensors & actuators", "🧬")
    banner("info", SWAT_NOTE)

    left, right = st.columns([1.5, 1])
    with left:
        section("Live snapshot", "Latest simulated MQTT reading")
        if latest:
            if severity == "CRITICAL":
                banner("bad", "<b>Critical anomaly</b> flagged in the live stream.")
            elif severity == "WARNING":
                banner("warn", "<b>Warning</b> condition in the live stream.")
            cc = st.columns(4)
            with cc[0]:
                metric_card("Power", f"{num(latest, 'power'):.3f} kW", "Electrical load", "⚡")
            with cc[1]:
                metric_card("Voltage", f"{num(latest, 'voltage'):.2f} V", "Line voltage", "🔌")
            with cc[2]:
                metric_card("Current", f"{num(latest, 'current'):.2f} A", "Line current", "〰️")
            with cc[3]:
                metric_card("Power Factor", f"{num(latest, 'power_factor'):.2f}", "Electrical quality", "📐")
        else:
            banner("warn", "No live MQTT telemetry found. Start <code>mqtt_publisher.py</code> and "
                           "<code>mqtt_subscriber.py</code>.")
    with right:
        section("SWaT Normal vs Attack", "Labelled attack-period sample")
        if not attack.empty and LABEL in attack:
            vc = attack[LABEL].value_counts()
            plot(donut(list(vc.index), list(vc.values), [SEV_COLOR["NORMAL"] if k == "Normal" else C["red"] for k in vc.index],
                       280, f"{len(attack):,}<br><span style='font-size:11px;color:#8b97b1'>rows</span>"))
        else:
            st.info("SWaT attack sample not loaded.")

    section("How the system works", "End-to-end detection pipeline")
    steps = [("IoT Devices", "Sensors & gateways publish telemetry"), ("MQTT Broker", "HiveMQ Cloud, TLS-secured transport"),
             ("Feature Extraction", "Electrical, process & network behaviour"), ("Isolation Forest", "Unsupervised anomaly scoring"),
             ("Alerts & Dashboard", "Severity, logs and live monitoring")]
    html = '<div class="flow">'
    for i, (t, d) in enumerate(steps, 1):
        html += f'<div class="flow-step"><span class="flow-num">{i}</span><b>{t}</b><span>{d}</span></div>'
        if i < len(steps):
            html += '<div class="flow-arrow">➜</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

# ============================================================
# LIVE MQTT DEMO
# ============================================================
elif mode == "Live MQTT Demo":
    section("🟢 Live MQTT telemetry", "Simulated telemetry only - no connection to a nuclear facility")
    if not latest:
        banner("warn", "No telemetry available. Start <code>mqtt_publisher.py</code> and <code>mqtt_subscriber.py</code>.")
    else:
        if severity == "CRITICAL":
            banner("bad", f"<b>Critical anomaly detected</b> on {latest.get('device_id', 'device')}.")
        elif severity == "WARNING":
            banner("warn", "<b>Warning condition</b> detected in telemetry.")
        else:
            banner("ok", "Live telemetry is within the normal operating envelope.")

        c = st.columns(4)
        with c[0]:
            metric_card("Power", f"{num(latest, 'power'):.3f} kW", "Electrical load", "⚡", sev_tone if anomaly_flag else "")
        with c[1]:
            metric_card("Voltage", f"{num(latest, 'voltage'):.2f} V", "Line voltage", "🔌")
        with c[2]:
            metric_card("Current", f"{num(latest, 'current'):.2f} A", "Line current", "〰️")
        with c[3]:
            metric_card("Power Factor", f"{num(latest, 'power_factor'):.2f}", "Electrical quality", "📐")

        extra = [(k, l, u, i) for k, l, u, i in [
            ("cooling_temp", "Cooling Temp", "°C", "🌡️"), ("system_pressure", "System Pressure", "", "🧭"),
            ("vibration", "Vibration", "", "📳"), ("radiation_index", "Radiation Index", "", "☢️"),
            ("message_rate", "Message Rate", "/s", "📡"), ("latency_ms", "Latency", "ms", "⏱️")] if k in latest]
        if extra:
            cols = st.columns(len(extra))
            for col, (k, l, u, i) in zip(cols, extra):
                with col:
                    metric_card(l, f"{num(latest, k):.2f} {u}".strip(), "Latest reading", i)

        section("Signal explorer", "Pick any signal and inspect its recent behaviour")
        if not hist.empty:
            sigs = [x for x in ["power", "voltage", "current", "power_factor", "cooling_temp", "system_pressure",
                                "vibration", "radiation_index", "message_rate", "latency_ms"] if x in hist.columns]
            if sigs:
                a1, a2 = st.columns([2, 1])
                sig = a1.selectbox("Live signal", sigs)
                smooth = a2.checkbox("Show moving average", value=True)
                h = hist.tail(300).copy()
                x = h["_t"] if "_t" in h.columns and h["_t"].notna().any() else h.index
                fig = go.Figure(line_trace(x, h[sig], sig, C["cyan"], fill=True))
                if smooth:
                    fig.add_trace(line_trace(x, pd.to_numeric(h[sig], errors="coerce").rolling(10, min_periods=1).mean(),
                                             "Moving avg (10)", C["amber"], dash="dot"))
                bad = h[h["_flag"]] if "_flag" in h.columns else pd.DataFrame()
                if not bad.empty:
                    bx = bad["_t"] if "_t" in bad.columns and bad["_t"].notna().any() else bad.index
                    fig.add_trace(go.Scatter(x=bx, y=pd.to_numeric(bad[sig], errors="coerce"), mode="markers",
                                             name="Flagged", marker=dict(size=9, color=C["red"], symbol="x",
                                                                          line=dict(width=1, color="#fff"))))
                fig.update_yaxes(title=sig)
                plot(make_chart(fig, 380))
        else:
            st.info("Waiting for telemetry history...")
        with st.expander("🔎 Raw telemetry record (JSON)"):
            st.json(latest)

# ============================================================
# SWaT RESEARCH MODE
# ============================================================
elif mode == "SWaT Research Mode":
    section("🔬 SWaT research mode", "Real industrial-control-system data analysed by the trained Isolation Forest")
    banner("info", SWAT_NOTE)
    if not swat_ready:
        swat_unavailable()
    else:
        try:
            model, features, scaler = swat_model()
            normal, attack = swat_data()
            gap(4)
            k1, k2, k3 = st.columns([1.2, 1.4, 1.6])
            kind_label = k1.selectbox("Dataset", ["Attack-period dataset", "Normal-operation dataset"])
            kind = "attack" if kind_label.startswith("Attack") else "normal"
            full = len(attack if kind == "attack" else normal)
            if full > 1000:
                size = k2.slider("Rows to analyse", 1000, min(20000, full), min(5000, full), 1000)
            else:
                size = full
                k2.caption(f"Using all {full:,} rows")
            data, dev = run_detection(kind, size)
            available = [f for f in features if f in data.columns]
            sensor = k3.selectbox("Process variable", available)

            n_an = int((data["AI Prediction"] == "ANOMALY").sum())
            c = st.columns(5)
            with c[0]:
                metric_card("Rows analysed", f"{len(data):,}", kind_label, "🧮")
            with c[1]:
                metric_card("Features", str(len(features)), "Model inputs", "🧬")
            with c[2]:
                metric_card("Known attacks", f"{int((attack[LABEL] == 'Attack').sum()):,}" if LABEL in attack else "—",
                            "In attack dataset", "🎯")
            with c[3]:
                metric_card("AI anomalies", f"{n_an:,}", f"{len(data) - n_an:,} normal", "🤖", "bad" if n_an else "ok")
            with c[4]:
                metric_card("Detected %", f"{100 * n_an / max(len(data), 1):.2f}%", "Share of rows flagged", "📊")

            section("Process timeline", f"{sensor} with AI-flagged points")
            show_labels = st.checkbox("Overlay ground-truth attack labels", value=True)
            d = data[["Timestamp", sensor, "AI Prediction"] + ([LABEL] if LABEL in data else [])].copy()
            x = d["Timestamp"] if d["Timestamp"].notna().any() else d.index
            fig = go.Figure(go.Scattergl(x=x, y=d[sensor], mode="lines", name=sensor,
                                         line=dict(width=1.6, color=C["cyan"])))
            if show_labels and LABEL in d:
                at = d[d[LABEL] == "Attack"].head(1500)
                if not at.empty:
                    fig.add_trace(go.Scattergl(x=at["Timestamp"] if d["Timestamp"].notna().any() else at.index,
                                               y=at[sensor], mode="markers", name="Labelled attack",
                                               marker=dict(size=5, color=C["amber"], opacity=.7)))
            an = d[d["AI Prediction"] == "ANOMALY"].head(1500)
            if not an.empty:
                fig.add_trace(go.Scattergl(x=an["Timestamp"] if d["Timestamp"].notna().any() else an.index,
                                           y=an[sensor], mode="markers", name="AI anomaly",
                                           marker=dict(size=6, color=C["red"], symbol="x")))
            fig.update_yaxes(title=sensor)
            plot(make_chart(fig, 400))

            m = performance(data)
            left, right = st.columns([1, 1])
            with left:
                section("AI result vs SWaT label", "Confusion matrix")
                if m:
                    plot(confusion_fig(m))
                else:
                    st.info("No ground-truth label column in this dataset.")
            with right:
                section("Which sensors drive the alarms?", "Mean |z-score| gap: flagged vs normal rows")
                if not dev.empty:
                    dv = dev.sort_values("Lift")
                    fig = go.Figure(go.Bar(x=dv["Lift"], y=dv["Sensor"], orientation="h",
                                           marker=dict(color=dv["Lift"], colorscale=[[0, "#4c1d95"], [1, "#22d3ee"]])))
                    fig.update_layout(hovermode="closest")
                    plot(make_chart(fig, 300))
                else:
                    st.info("Not enough flagged/normal rows to compare sensors.")
            if m and kind == "attack":
                perf_cards(m)

            section("Detection results", "Most anomalous rows first (top 100)")
            cols = ["Timestamp", "AI Prediction", "Anomaly Score"] + ([LABEL] if LABEL in data else []) + [sensor]
            show_table(style_predictions(data.sort_values("Anomaly Score")[cols].head(100)))
        except Exception as e:
            st.error(f"SWaT analysis failed: {e}")

# ============================================================
# AI ANOMALY DETECTION
# ============================================================
elif mode == "AI Anomaly Detection":
    section("🤖 AI anomaly detection", "Isolation Forest scoring on the SWaT attack-period dataset")
    if not swat_ready:
        swat_unavailable()
    else:
        try:
            model, features, scaler = swat_model()
            _, attack = swat_data()
            data, dev = run_detection("attack", min(10000, len(attack)))
            n_an = int((data["AI Prediction"] == "ANOMALY").sum())
            c = st.columns(4)
            with c[0]:
                metric_card("Samples scored", f"{len(data):,}", "From attack dataset", "🧮")
            with c[1]:
                metric_card("AI anomalies", f"{n_an:,}", "Flagged by Isolation Forest", "🤖", "bad" if n_an else "ok")
            with c[2]:
                metric_card("Anomaly rate", f"{100 * n_an / len(data):.2f}%", "Share of samples", "📊")
            with c[3]:
                est = getattr(model, "n_estimators", "—")
                metric_card("Model", "Isolation Forest", f"{est} trees • {len(features)} features", "🌲")

            m = performance(data)
            if m:
                gap(2)
                perf_cards(m)

            left, right = st.columns([1.5, 1])
            with left:
                section("Anomaly score distribution", "Scores below 0 are classified as anomalies")
                fig = go.Figure()
                for lab, colr in [("NORMAL", C["green"]), ("ANOMALY", C["red"])]:
                    s = data.loc[data["AI Prediction"] == lab, "Anomaly Score"]
                    fig.add_trace(go.Histogram(x=s, name=lab, nbinsx=60, marker_color=colr, opacity=.75))
                fig.add_vline(x=0, line_dash="dash", line_color="#f1f5f9", annotation_text="decision boundary")
                fig.update_layout(barmode="overlay", hovermode="closest")
                plot(make_chart(fig, 360))
            with right:
                section("Confusion matrix", "AI vs SWaT label")
                if m:
                    plot(confusion_fig(m))
                else:
                    st.info("No label column available.")

            section("Top deviating sensors", "Sensors that move furthest from normal when the AI raises an alarm")
            if not dev.empty:
                dv = dev.sort_values("Lift")
                fig = go.Figure()
                fig.add_trace(go.Bar(x=dv["Normal"], y=dv["Sensor"], orientation="h", name="Normal rows", marker_color=C["green"]))
                fig.add_trace(go.Bar(x=dv["Anomalous"], y=dv["Sensor"], orientation="h", name="Flagged rows", marker_color=C["red"]))
                fig.update_layout(barmode="group", hovermode="closest")
                plot(make_chart(fig, 380))

            section("Most anomalous samples", "Top 100 by anomaly score")
            cols = ["Timestamp", LABEL, "AI Prediction", "Anomaly Score"] if LABEL in data else ["Timestamp", "AI Prediction", "Anomaly Score"]
            show_table(style_predictions(data.sort_values("Anomaly Score")[cols].head(100)))
        except Exception as e:
            st.error(f"AI analysis failed: {e}")

# ============================================================
# CYBERSECURITY MONITOR
# ============================================================
elif mode == "Cybersecurity Monitor":
    section("🛡️ Cybersecurity monitor", "Indicators from simulated MQTT telemetry and SWaT research data")
    crit = int(alerts["severity"].astype(str).str.upper().eq("CRITICAL").sum()) if "severity" in alerts else 0
    c = st.columns(4)
    with c[0]:
        metric_card("Recorded alerts", f"{len(alerts):,}", "MQTT alert log", "🗂️", "bad" if len(alerts) else "ok")
    with c[1]:
        metric_card("Critical", f"{crit:,}", "Critical-severity events", "🔴", "bad" if crit else "ok")
    with c[2]:
        metric_card("MQTT link", "CONNECTED" if latest else "WAITING", "Telemetry stream", "🔗", "ok" if latest else "warn")
    with c[3]:
        metric_card("Current state", severity, "Latest telemetry", "📍", sev_tone)

    left, right = st.columns([1, 1.3])
    with left:
        section("Security signals", "Live health of each indicator")
        if latest:
            mr, pl, lt, au = (num(latest, k) for k in ("message_rate", "payload_size", "latency_ms", "auth_failures"))
            signal_row("Traffic rate", "ELEVATED" if mr > 5 else "NORMAL", f"{mr:.2f} msg/s (alert > 5)", mr / 10 * 100, "warn" if mr > 5 else "ok")
            signal_row("Payload size", "ELEVATED" if pl > 2000 else "NORMAL", f"{pl:.0f} B (alert > 2000)", pl / 4000 * 100, "warn" if pl > 2000 else "ok")
            signal_row("Authentication", "ATTENTION" if au > 0 else "NORMAL", f"{au:.0f} failed attempts", au * 20, "bad" if au > 0 else "ok")
            signal_row("Latency", "ELEVATED" if lt > 300 else "NORMAL", f"{lt:.1f} ms (alert > 300)", lt / 600 * 100, "warn" if lt > 300 else "ok")
        else:
            st.info("Start the MQTT publisher/subscriber to see live security signals.")
    with right:
        section("SWaT attack density over time", "Share of labelled attack rows per time bin")
        try:
            if not swat_files_ok:
                raise FileNotFoundError("SWaT dataset not found")
            _, a = swat_data()
            d = a[["Timestamp", LABEL]].copy()
            d["Timestamp"] = pd.to_datetime(d["Timestamp"].astype(str).str.strip(), errors="coerce", dayfirst=True)
            d = d.dropna().sort_values("Timestamp")
            d["bin"] = pd.qcut(np.arange(len(d)), q=min(60, max(len(d) // 20, 2)), labels=False, duplicates="drop")
            g = d.groupby("bin").agg(t=("Timestamp", "first"), rate=(LABEL, lambda s: (s == "Attack").mean() * 100))
            fig = go.Figure(go.Scatter(x=g["t"], y=g["rate"], mode="lines", fill="tozeroy",
                                       line=dict(color=C["red"], width=2.2, shape="spline"),
                                       fillcolor=hex_to_rgba(C["red"], .12), name="Attack %"))
            fig.update_yaxes(title="% attack rows", range=[0, 100])
            plot(make_chart(fig, 300))
        except Exception as e:
            st.info(f"SWaT chart unavailable: {e}")

    l2, r2 = st.columns([1, 1.3])
    with l2:
        section("SWaT label distribution")
        try:
            _, a = swat_data()
            vc = a[LABEL].value_counts()
            plot(donut(list(vc.index), list(vc.values), [C["green"] if k == "Normal" else C["red"] for k in vc.index], 290))
        except Exception:
            st.info("SWaT labels unavailable.")
    with r2:
        section("Recent MQTT alerts", "Latest 100 events")
        if alerts.empty:
            banner("ok", "No recorded MQTT alerts currently available.")
        else:
            show_table(alerts.tail(100).iloc[::-1])

# ============================================================
# ALERTS & LOGS
# ============================================================
else:
    section("🚨 Alerts & logs", "Recorded anomaly and cybersecurity events")
    if alerts.empty:
        banner("ok", "No alert records found.")
    else:
        sv = alerts["severity"].astype(str).str.upper() if "severity" in alerts else pd.Series(dtype=str)
        c = st.columns(4)
        with c[0]:
            metric_card("Total alerts", f"{len(alerts):,}", "Stored events", "🗂️")
        with c[1]:
            metric_card("Critical", f"{int(sv.eq('CRITICAL').sum()):,}", "Critical events", "🔴", "bad" if sv.eq("CRITICAL").any() else "ok")
        with c[2]:
            metric_card("Warnings", f"{int(sv.eq('WARNING').sum()):,}", "Warning events", "🟡", "warn" if sv.eq("WARNING").any() else "ok")
        with c[3]:
            metric_card("Current state", severity, "Latest telemetry", "📍", sev_tone)

        left, right = st.columns([1.9, 1])
        with right:
            if len(sv):
                section("Severity mix")
                vc = sv.value_counts()
                plot(donut(list(vc.index), list(vc.values), [SEV_COLOR.get(k, C["blue"]) for k in vc.index], 290,
                           f"{len(alerts):,}"))
            if "attack_mode" in alerts:
                section("Top attack modes")
                top = alerts["attack_mode"].astype(str).str.replace("_", " ").str.title().value_counts().head(6)
                fig = go.Figure(go.Bar(x=top.values[::-1], y=top.index[::-1], orientation="h", marker=dict(color=C["violet"])))
                fig.update_layout(hovermode="closest")
                plot(make_chart(fig, 240))
        with left:
            section("Event log", "Newest 200 events")
            tbl = alerts.tail(200).iloc[::-1]
            if "severity" in tbl:
                def colour(v):
                    c_ = SEV_COLOR.get(str(v).upper())
                    return f"color:{c_};font-weight:700" if c_ else ""
                s = tbl.style
                show_table((s.map if hasattr(s, "map") else s.applymap)(colour, subset=["severity"]))
            else:
                show_table(tbl)
            st.download_button("⬇ Download alert log CSV", alerts.to_csv(index=False).encode("utf-8"),
                               "nuclear_iot_alerts.csv", "text/csv")

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f"""<div class="footer">{PROJECT_NAME} • Educational project • SWaT = real ICS research dataset •
    MQTT = simulated telemetry<br>Not connected to real nuclear infrastructure • {TEAM_NAME}</div>""",
    unsafe_allow_html=True)

# fallback auto-refresh when streamlit-autorefresh is not installed (live pages only)
if auto and st_autorefresh is None and mode in LIVE_PAGES:
    time.sleep(refresh_s)
    st.rerun()