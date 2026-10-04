import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


NORMAL_PATH = "data/swat/SWaT_Dataset_Normal_v1.xlsx"
ATTACK_PATH = "data/swat/SWaT_Dataset_Attack_v0.xlsx"

MODEL_PATH = "models/swat_anomaly_model.joblib"
SCALER_PATH = "models/swat_scaler.joblib"


print("Loading SWaT normal data...")
normal = pd.read_excel(NORMAL_PATH, header=1)

print("Loading SWaT attack data...")
attack = pd.read_excel(ATTACK_PATH, header=1)

print("Normal shape:", normal.shape)
print("Attack shape:", attack.shape)


# --------------------------------------------------
# CLEAN COLUMN NAMES
# --------------------------------------------------

normal.columns = normal.columns.str.strip()
attack.columns = attack.columns.str.strip()


# --------------------------------------------------
# CLEAN LABELS
# --------------------------------------------------

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


print("\nAttack labels after cleaning:")
print(attack["Normal/Attack"].value_counts())


# --------------------------------------------------
# REMOVE TIMESTAMP AND LABEL
# --------------------------------------------------

normal_features = normal.drop(columns=["Timestamp", "Normal/Attack"], errors="ignore")
attack_features = attack.drop(columns=["Timestamp", "Normal/Attack"], errors="ignore")


# --------------------------------------------------
# KEEP ONLY NUMERIC SENSOR DATA
# --------------------------------------------------

normal_features = normal_features.select_dtypes(include=np.number)
attack_features = attack_features.select_dtypes(include=np.number)


# Make sure both datasets have identical columns
common_columns = normal_features.columns.intersection(
    attack_features.columns
)

normal_features = normal_features[common_columns]
attack_features = attack_features[common_columns]


print("\nNumber of sensor features:", len(common_columns))
print("Features:")
print(list(common_columns))


# --------------------------------------------------
# HANDLE MISSING / INFINITE VALUES
# --------------------------------------------------

normal_features = normal_features.replace([np.inf, -np.inf], np.nan)
attack_features = attack_features.replace([np.inf, -np.inf], np.nan)

normal_features = normal_features.fillna(normal_features.median())
attack_features = attack_features.fillna(normal_features.median())


# --------------------------------------------------
# SAMPLE DATA FOR 8 GB RAM LAPTOP
# --------------------------------------------------

NORMAL_SAMPLE_SIZE = 100000
ATTACK_SAMPLE_SIZE = 100000

normal_sample = normal_features.sample(
    n=min(NORMAL_SAMPLE_SIZE, len(normal_features)),
    random_state=42
)

attack_sample_indices = np.random.RandomState(42).choice(
    len(attack_features),
    size=min(ATTACK_SAMPLE_SIZE, len(attack_features)),
    replace=False
)

attack_sample = attack_features.iloc[attack_sample_indices]

attack_labels = attack["Normal/Attack"].iloc[
    attack_sample_indices
].reset_index(drop=True)

normal_sample = normal_sample.reset_index(drop=True)
attack_sample = attack_sample.reset_index(drop=True)


print("\nTraining normal samples:", len(normal_sample))
print("Testing attack samples:", len(attack_sample))


# --------------------------------------------------
# SCALE FEATURES
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(normal_sample)


# --------------------------------------------------
# TRAIN ISOLATION FOREST
# --------------------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=150,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train)

print("Training complete.")


# --------------------------------------------------
# TEST ON NORMAL DATA
# --------------------------------------------------

normal_predictions = model.predict(
    scaler.transform(normal_sample)
)


# --------------------------------------------------
# TEST ON ATTACK DATA
# --------------------------------------------------

attack_predictions = model.predict(
    scaler.transform(attack_sample)
)


# --------------------------------------------------
# COMBINE RESULTS
# --------------------------------------------------

y_true = np.concatenate([
    np.zeros(len(normal_sample)),
    np.ones(len(attack_sample))
])


y_pred = np.concatenate([
    (normal_predictions == -1).astype(int),
    (attack_predictions == -1).astype(int)
])


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

print("\n==============================")
print("SWaT MODEL EVALUATION")
print("==============================")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model_package = {
    "model": model,
    "features": list(common_columns),
    "dataset": "SWaT",
    "model_type": "Isolation Forest"
}

joblib.dump(model_package, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)


print("\n==============================")
print("MODEL SAVED")
print("==============================")

print("Model:", MODEL_PATH)
print("Scaler:", SCALER_PATH)

print("\nSWaT training completed successfully.")