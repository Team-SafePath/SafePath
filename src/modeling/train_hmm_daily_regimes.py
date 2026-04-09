from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as e:
    raise ImportError(
        "hmmlearn is required. Install it with: pip install hmmlearn"
    ) from e


INPUT_PATH = Path("data/processed/hmm_daily_features_with_infra.csv")
OUTPUT_STATES_PATH = Path("data/processed/hmm_daily_states_with_infra.csv")
OUTPUT_SUMMARY_PATH = Path("data/processed/hmm_state_summary_with_infra.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "hmm_daily_regimes_with_infra.joblib"
SCALER_PATH = MODEL_DIR / "hmm_daily_regimes_with_infra_scaler.joblib"
METRICS_PATH = MODEL_DIR / "hmm_daily_regimes_with_infra_metrics.json"

N_STATES = 3
RANDOM_STATE = 42


def build_state_label_map(summary_df: pd.DataFrame):
    ordered = summary_df.sort_values("avg_total_crashes", ascending=False).reset_index(drop=True)

    labels = {}
    for i, row in ordered.iterrows():
        state_id = int(row["hidden_state"])
        if i == 0:
            labels[state_id] = "High Risk Regime"
        elif i == 1:
            labels[state_id] = "Moderate Risk Regime"
        else:
            labels[state_id] = "Low Risk Regime"
    return labels


def main():
    print("Loading daily HMM dataset...")
    df = pd.read_csv(INPUT_PATH)

    required_cols = {
        "date",
        "total_crashes",
        "crash_rate",
        "avg_crashes_last_7_days",
        "avg_crashes_last_30_days",
        "temperature_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
        "rain_indicator",
        "crashes_7d_avg_city",
        "crashes_30d_avg_city",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in HMM input: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    feature_cols = [
        "total_crashes",
        "crash_rate",
        "avg_crashes_last_7_days",
        "avg_crashes_last_30_days",
        "temperature_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
        "rain_indicator",
        "crashes_7d_avg_city",
        "crashes_30d_avg_city",
    ]

    X = df[feature_cols].copy()

    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric HMM columns found: {non_numeric}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nTraining Gaussian HMM...")
    hmm = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=300,
        random_state=RANDOM_STATE,
    )
    hmm.fit(X_scaled)

    hidden_states = hmm.predict(X_scaled)
    state_probs = hmm.predict_proba(X_scaled).max(axis=1)

    result = df.copy()
    result["hidden_state"] = hidden_states
    result["state_probability_max"] = state_probs

    summary = (
        result.groupby("hidden_state")
        .agg(
            n_days=("date", "count"),
            avg_total_crashes=("total_crashes", "mean"),
            avg_crash_rate=("crash_rate", "mean"),
            avg_crashes_last_7_days=("avg_crashes_last_7_days", "mean"),
            avg_crashes_last_30_days=("avg_crashes_last_30_days", "mean"),
            avg_temperature_2m_mean=("temperature_2m_mean", "mean"),
            avg_precipitation_sum=("precipitation_sum", "mean"),
            avg_windspeed_10m_max=("windspeed_10m_max", "mean"),
            avg_rain_indicator=("rain_indicator", "mean"),
        )
        .reset_index()
    )

    label_map = build_state_label_map(summary)
    result["state_label"] = result["hidden_state"].map(label_map)
    summary["state_label"] = summary["hidden_state"].map(label_map)

    transition_matrix = hmm.transmat_.tolist()
    state_counts = result["hidden_state"].value_counts().sort_index().to_dict()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_STATES_PATH, index=False)
    summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)
    joblib.dump(hmm, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics = {
        "input_path": str(INPUT_PATH),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "n_states": int(N_STATES),
        "state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "transition_matrix": transition_matrix,
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "states_path": str(OUTPUT_STATES_PATH),
        "summary_path": str(OUTPUT_SUMMARY_PATH),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved states to:", OUTPUT_STATES_PATH)
    print("Saved summary to:", OUTPUT_SUMMARY_PATH)
    print("Saved metrics to:", METRICS_PATH)

    print("\nState counts:")
    print(result["hidden_state"].value_counts().sort_index())

    print("\nState summary:")
    print(summary)

    print("\nTransition matrix:")
    for row in transition_matrix:
        print([round(x, 6) for x in row])


if __name__ == "__main__":
    main()