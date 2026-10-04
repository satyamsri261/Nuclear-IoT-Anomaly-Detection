import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="SWaT Research Dashboard",
    page_icon="🔬",
    layout="wide"
)

NORMAL_PATH = "data/swat/SWaT_Dataset_Normal_v1.xlsx"
ATTACK_PATH = "data/swat/SWaT_Dataset_Attack_v0.xlsx"

MODEL_PATH = "models/swat_anomaly_model.joblib"
SCALER_PATH = "models/swat_scaler.joblib"


st.title("🔬 SWaT Industrial Cybersecurity Research")
st.caption(
    "Real industrial control system dataset • AI anomaly detection research"
)

st.info(
    "SWaT is a real industrial control-system testbed dataset. "
    "It is not nuclear-plant telemetry."
)


@st.cache_resource
def load_model():
    package = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return package["model"], package["features"], scaler


@st.cache_data
def load_swat_data():

    normal = pd.read_excel(
        NORMAL_PATH,
        header=1
    )

    attack = pd.read_excel(
        ATTACK_PATH,
        header=1
    )

    normal.columns = normal.columns.str.strip()
    attack.columns = attack.columns.str.strip()

    attack["Normal/Attack"] = (
        attack["Normal/Attack"]
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.lower()
    )

    attack["Normal/Attack"] = attack["Normal/Attack"].replace({
        "normal": "Normal",
        "attack": "Attack"
    })

    return normal, attack


try:

    model, features, scaler = load_model()

    normal, attack = load_swat_data()

except Exception as e:

    st.error(f"Could not load SWaT data/model: {e}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("🔬 Research Controls")

dataset_type = st.sidebar.selectbox(
    "Dataset",
    ["Normal Operation", "Attack Dataset"]
)

sample_size = st.sidebar.slider(
    "Rows to display",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000
)


# --------------------------------------------------
# SELECT DATA
# --------------------------------------------------

if dataset_type == "Normal Operation":

    data = normal.sample(
        n=min(sample_size, len(normal)),
        random_state=42
    ).copy()

else:

    data = attack.sample(
        n=min(sample_size, len(attack)),
        random_state=42
    ).copy()


# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Dataset Rows",
        f"{len(normal) + len(attack):,}"
    )

with col2:
    st.metric(
        "Sensor Features",
        len(features)
    )

with col3:
    attack_count = (attack["Normal/Attack"] == "Attack").sum()

    st.metric(
        "Attack Records",
        f"{attack_count:,}"
    )

with col4:
    normal_count = (attack["Normal/Attack"] == "Normal").sum()

    st.metric(
        "Normal Records",
        f"{normal_count:,}"
    )


# --------------------------------------------------
# LABEL DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Dataset Distribution")

label_counts = attack["Normal/Attack"].value_counts().reset_index()

label_counts.columns = ["Status", "Count"]

fig = px.bar(
    label_counts,
    x="Status",
    y="Count",
    title="Normal vs Attack Records"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# SENSOR SELECTION
# --------------------------------------------------

st.subheader("📈 Sensor Analysis")

available_features = [
    f for f in features
    if f in data.columns
]

selected_sensor = st.selectbox(
    "Select an industrial sensor/process variable",
    available_features
)


# --------------------------------------------------
# SENSOR GRAPH
# --------------------------------------------------

plot_data = data[
    ["Timestamp", selected_sensor]
].copy()

plot_data["Timestamp"] = pd.to_datetime(
    plot_data["Timestamp"],
    errors="coerce"
)

fig_sensor = px.line(
    plot_data,
    x="Timestamp",
    y=selected_sensor,
    title=f"{selected_sensor} over Time"
)

st.plotly_chart(
    fig_sensor,
    use_container_width=True
)


# --------------------------------------------------
# ATTACK TIMELINE
# --------------------------------------------------

if dataset_type == "Attack Dataset":

    st.subheader("🚨 Attack Timeline")

    timeline = data[
        ["Timestamp", "Normal/Attack"]
    ].copy()

    timeline["Timestamp"] = pd.to_datetime(
        timeline["Timestamp"],
        errors="coerce"
    )

    timeline["Attack"] = (
        timeline["Normal/Attack"] == "Attack"
    ).astype(int)

    fig_attack = px.scatter(
        timeline,
        x="Timestamp",
        y="Attack",
        color="Normal/Attack",
        title="Attack / Normal Timeline"
    )

    st.plotly_chart(
        fig_attack,
        use_container_width=True
    )


# --------------------------------------------------
# AI ANOMALY DETECTION
# --------------------------------------------------

st.subheader("🤖 AI Anomaly Detection")

model_data = data[features].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.fillna(0)

predictions = model.predict(
    scaler.transform(model_data)
)

data["AI Prediction"] = np.where(
    predictions == -1,
    "ANOMALY",
    "NORMAL"
)

anomaly_count = (
    data["AI Prediction"] == "ANOMALY"
).sum()

normal_prediction_count = (
    data["AI Prediction"] == "NORMAL"
).sum()


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "AI Anomalies",
        f"{anomaly_count:,}"
    )

with col2:

    st.metric(
        "AI Normal",
        f"{normal_prediction_count:,}"
    )

with col3:

    anomaly_percentage = (
        anomaly_count / len(data)
    ) * 100

    st.metric(
        "Anomaly %",
        f"{anomaly_percentage:.2f}%"
    )


# --------------------------------------------------
# AI RESULT TABLE
# --------------------------------------------------

st.subheader("🔎 AI Detection Results")

result_columns = [
    "Timestamp",
    "AI Prediction"
]

if "Normal/Attack" in data.columns:

    result_columns.append(
        "Normal/Attack"
    )

result_columns.append(
    selected_sensor
)

st.dataframe(
    data[result_columns].head(100),
    use_container_width=True
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Educational research dashboard | SWaT industrial control-system dataset"
)