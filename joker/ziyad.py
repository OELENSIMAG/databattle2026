# ============================================================
# Meteorage baseline: 25 engineered features + LightGBM
# Goal:
#   Predict whether there will be NO lightning within 20 km
#   of the airport in the next HORIZON_MIN minutes.
#
# Label:
#   y = 1  -> no future strike within 20 km in next horizon
#   y = 0  -> at least one future strike within 20 km in next horizon
#
# Recommended split:
#   time-based split, never random row split
# ============================================================

import numpy as np
import pandas as pd # type: ignore
from lightgbm import LGBMClassifier # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

# -----------------------------
# Config
# -----------------------------
HORIZON_MIN = 20
INSIDE_RADIUS_KM = 20

# -----------------------------
# Load and clean
# -----------------------------
df = pd.read_csv("/Users/oukhtite/3A/data_train_databattle2026/databattle2026/data_train_databattle2026/segment_alerts_all_airports_train.csv")

df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values(["airport", "date"]).reset_index(drop=True)

# Clean booleans
for col in ["icloud", "is_last_lightning_cloud_ground"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map({
            "true": 1, "false": 0, "1": 1, "0": 0, "nan": np.nan
        })

# Keep a binary "inside operational zone"
df["inside20"] = (df["dist"] <= INSIDE_RADIUS_KM).astype(int)

# Useful numeric cleanup
num_cols = ["dist", "amplitude", "azimuth", "maxis"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# Feature engineering helpers
# -----------------------------
def slope_from_series(y):
    """Linear slope over equally spaced index positions."""
    y = pd.Series(y).dropna().values
    n = len(y)
    if n < 2:
        return np.nan
    x = np.arange(n)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return np.nan
    return np.sum((x - x_mean) * (y - y_mean)) / denom

def circular_std_deg(angles_deg):
    """Circular std for azimuth-like angles in degrees."""
    a = pd.Series(angles_deg).dropna().values
    if len(a) == 0:
        return np.nan
    radians = np.deg2rad(a)
    C = np.mean(np.cos(radians))
    S = np.mean(np.sin(radians))
    R = np.sqrt(C**2 + S**2)
    if R <= 1e-12:
        return 180.0
    return np.rad2deg(np.sqrt(-2 * np.log(R)))

def add_group_time_features(g):
    g = g.sort_values("date").copy()

    # Basic event-to-event timing
    g["gap_min"] = g["date"].diff().dt.total_seconds() / 60.0
    g["gap_prev_1"] = g["gap_min"].shift(1)
    g["gap_prev_2"] = g["gap_min"].shift(2)

    # Rolling statistics on gaps (last 5 observed gaps, excluding current row's future)
    g["gap_mean_5"] = g["gap_min"].rolling(5, min_periods=2).mean()
    g["gap_std_5"] = g["gap_min"].rolling(5, min_periods=2).std()
    g["gap_max_5"] = g["gap_min"].rolling(5, min_periods=2).max()
    g["gap_min_5"] = g["gap_min"].rolling(5, min_periods=2).min()

    # Gap dynamics
    g["gap_ratio"] = g["gap_min"] / g["gap_prev_1"]
    g["gap_accel"] = g["gap_min"] - g["gap_prev_1"]
    g["gap_trend_5"] = g["gap_min"].rolling(5, min_periods=3).apply(slope_from_series, raw=False)

    # Rolling event counts based on time windows
    # Count recent strikes up to and including current row
    t = g.set_index("date")
    g["strikes_5m"] = t["lightning_id"].rolling("5min").count().values
    g["strikes_10m"] = t["lightning_id"].rolling("10min").count().values
    g["strikes_30m"] = t["lightning_id"].rolling("30min").count().values

    # Inside-20 km event counts
    g["inside20_10m"] = t["inside20"].rolling("10min").sum().values
    g["inside20_30m"] = t["inside20"].rolling("30min").sum().values

    # Outer ring counts
    ring20_40 = ((g["dist"] > 20) & (g["dist"] <= 40)).astype(int)
    ring40_80 = ((g["dist"] > 40) & (g["dist"] <= 80)).astype(int)
    t_ring = pd.DataFrame(
        {"r20_40": ring20_40.values, "r40_80": ring40_80.values},
        index=g["date"]
    )
    g["ring20_40_10m"] = t_ring["r20_40"].rolling("10min").sum().values
    g["ring40_80_10m"] = t_ring["r40_80"].rolling("10min").sum().values

    # Distance features
    g["last_distance"] = g["dist"]
    g["mean_distance_5"] = g["dist"].rolling(5, min_periods=2).mean()
    g["min_distance_5"] = g["dist"].rolling(5, min_periods=2).min()
    g["distance_slope_5"] = g["dist"].rolling(5, min_periods=3).apply(slope_from_series, raw=False)
    g["distance_delta"] = g["dist"].diff()

    # Approx movement speed (km/min between consecutive lightning distances)
    g["radial_speed"] = g["distance_delta"] / g["gap_min"]

    # Amplitude features
    g["mean_amp_5"] = g["amplitude"].rolling(5, min_periods=2).mean()
    g["max_amp_5"] = g["amplitude"].rolling(5, min_periods=2).max()
    g["std_amp_5"] = g["amplitude"].rolling(5, min_periods=2).std()
    g["amp_slope_5"] = g["amplitude"].rolling(5, min_periods=3).apply(slope_from_series, raw=False)

    # Cloud / type proxy
    if "icloud" in g.columns:
        g["icloud_frac_10m"] = t["icloud"].rolling("10min").mean().values
    else:
        g["icloud_frac_10m"] = np.nan

    # Azimuth structure
    g["azimuth_std_5"] = g["azimuth"].rolling(5, min_periods=3).apply(circular_std_deg, raw=False)

    # Time since last inside-20 event
    inside_dates = g["date"].where(g["inside20"] == 1)
    g["last_inside20_date"] = inside_dates.ffill()
    g["time_since_last_inside20_min"] = (
        (g["date"] - g["last_inside20_date"]).dt.total_seconds() / 60.0
    )

    # Storm age proxy:
    # define a rough reconstructed episode using 60-min inactivity per airport
    new_episode = (g["gap_min"].isna()) | (g["gap_min"] > 60)
    g["episode_id"] = new_episode.cumsum()
    episode_start = g.groupby("episode_id")["date"].transform("min")
    g["storm_age_min"] = (g["date"] - episode_start).dt.total_seconds() / 60.0

    # Context
    g["hour"] = g["date"].dt.hour
    g["month"] = g["date"].dt.month

    return g

df = df.groupby("airport", group_keys=False).apply(add_group_time_features).reset_index(drop=True)

# -----------------------------
# Build target
# y = 1 if NO future lightning within 20km in next HORIZON_MIN minutes
# -----------------------------
def add_target(g, horizon_min=30):
    g = g.sort_values("date").copy()
    times = g["date"].values.astype("datetime64[ns]")
    inside = g["inside20"].values

    future_inside_within_horizon = np.zeros(len(g), dtype=int)

    # O(n^2) but simple and readable; optimize later if needed
    for i in range(len(g)):
        t0 = times[i]
        upper = t0 + np.timedelta64(horizon_min, "m")
        mask = (times > t0) & (times <= upper) & (inside == 1)
        future_inside_within_horizon[i] = int(mask.any())

    g["target_no_inside20_next30"] = 1 - future_inside_within_horizon
    return g

df = df.groupby("airport", group_keys=False).apply(add_target, horizon_min=HORIZON_MIN).reset_index(drop=True)

# -----------------------------
# Recommended 25 features
# -----------------------------
FEATURES = [
    # Temporal / gap (8)
    "gap_min",
    "gap_prev_1",
    "gap_prev_2",
    "gap_mean_5",
    "gap_std_5",
    "gap_ratio",
    "gap_accel",
    "gap_trend_5",

    # Activity (3)
    "strikes_5m",
    "strikes_10m",
    "strikes_30m",

    # Spatial / distance (5)
    "last_distance",
    "mean_distance_5",
    "min_distance_5",
    "distance_slope_5",
    "radial_speed",

    # Operational zone / rings (4)
    "inside20_10m",
    "inside20_30m",
    "time_since_last_inside20_min",
    "ring20_40_10m",

    # Outside context (1)
    "ring40_80_10m",

    # Intensity (3)
    "mean_amp_5",
    "max_amp_5",
    "amp_slope_5",

    # Context (1 kept here, plus 2 more below makes 25 total)
    "icloud_frac_10m",
    "hour",
    "month",
]

# That's 27; trim to the recommended 25
FEATURES = [
    "gap_min",
    "gap_prev_1",
    "gap_prev_2",
    "gap_mean_5",
    "gap_std_5",
    "gap_ratio",
    "gap_accel",
    "gap_trend_5",
    "strikes_5m",
    "strikes_10m",
    "strikes_30m",
    "last_distance",
    "mean_distance_5",
    "min_distance_5",
    "distance_slope_5",
    "radial_speed",
    "inside20_10m",
    "inside20_30m",
    "time_since_last_inside20_min",
    "ring20_40_10m",
    "ring40_80_10m",
    "mean_amp_5",
    "max_amp_5",
    "hour",
    "month",
]

TARGET = "target_no_inside20_next30"

# -----------------------------
# Train / validation split
# IMPORTANT: time split, not random split
# -----------------------------
# Example: last 20% by time as validation
cutoff = df["date"].quantile(0.80)

train_df = df[df["date"] < cutoff].copy()
valid_df = df[df["date"] >= cutoff].copy()

# Drop rows with missing target / too-early rolling features
train_df = train_df.dropna(subset=FEATURES + [TARGET]).copy()
valid_df = valid_df.dropna(subset=FEATURES + [TARGET]).copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(int)

X_valid = valid_df[FEATURES]
y_valid = valid_df[TARGET].astype(int)

# -----------------------------
# LightGBM model
# -----------------------------
model = LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    class_weight="balanced",
    verbose=10
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="binary_logloss",
)

# -----------------------------
# Predictions
# -----------------------------
valid_proba = model.predict_proba(X_valid)[:, 1]   # probability of NO inside-20 strike in next 30 min
valid_pred = (valid_proba >= 0.5).astype(int)

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_classification(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Probabilistic metrics
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    ap = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_proba)

    cm = confusion_matrix(y_true, y_pred)

    print("=== Standard metrics ===")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1-score       : {f1:.4f}")
    print(f"ROC-AUC        : {auc:.4f}")
    print(f"PR-AUC         : {ap:.4f}")
    print(f"Brier score    : {brier:.4f}")
    print()
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    # Safety-focused interpretation
    # Positive class = "safe to stop alert"
    # So FP = model says safe, but actually another inside-20 strike happens => dangerous false clear
    tn, fp, fn, tp = cm.ravel()
    false_clear_rate = fp / (fp + tp) if (fp + tp) > 0 else np.nan
    missed_stop_rate = fn / (fn + tn) if (fn + tn) > 0 else np.nan

    print()
    print("=== Operational metrics ===")
    print(f"False clear rate (dangerous) : {false_clear_rate:.4f}")
    print(f"Missed stop rate             : {missed_stop_rate:.4f}")

evaluate_classification(y_valid, valid_pred, valid_proba)

# -----------------------------
# Threshold tuning for safety
# -----------------------------
# In this task, 0.5 is often NOT the right threshold.
# Higher threshold => safer, but fewer alerts ended early.
thresholds = np.arange(0.50, 0.96, 0.05)

rows = []
for th in thresholds:
    pred = (valid_proba >= th).astype(int)
    cm = confusion_matrix(y_valid, pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_valid, pred, zero_division=0)
    recall = recall_score(y_valid, pred, zero_division=0)

    false_clear_rate = fp / (fp + tp) if (fp + tp) > 0 else np.nan
    stop_rate = pred.mean()  # how often you actually stop alerts

    rows.append({
        "threshold": th,
        "precision_safe_stop": precision,
        "recall_safe_stop": recall,
        "false_clear_rate": false_clear_rate,
        "stop_rate": stop_rate,
    })

threshold_table = pd.DataFrame(rows)
print("\n=== Threshold analysis ===")
print(threshold_table.to_string(index=False))

# -----------------------------
# Feature importance
# -----------------------------
feat_imp = (
    pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\n=== Top feature importances ===")
print(feat_imp.head(15).to_string(index=False))

input("\nPress Enter to exit...")