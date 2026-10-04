"""Train an Isolation Forest on synthetic normal facility telemetry."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from energy_simulator import generate_telemetry
from config import MODEL_FILE, HISTORY_FILE

FEATURES = [
    "voltage", "current", "power", "power_factor",
    "cooling_temp", "system_pressure", "vibration", "radiation_index",
    "message_rate", "payload_size", "latency_ms", "auth_failures",
]


def make_dataset(n=3000):
    rows = [generate_telemetry(attack_mode="normal") for _ in range(n)]
    return pd.DataFrame(rows)


def main():
    df = make_dataset()
    X = df[FEATURES]
    model = IsolationForest(
        n_estimators=250,
        contamination=0.03,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_FILE)
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    df.to_csv("data/training_normal_data.csv", index=False)

    print("Model trained successfully")
    print("Features:", FEATURES)
    print("Model:", MODEL_FILE)


if __name__ == "__main__":
    main()
