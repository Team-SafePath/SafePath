from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as e:
    raise ImportError(
        "hmmlearn is not installed. Run: pip install hmmlearn"
    ) from e


INPUT_PATH = Path("data/processed/full_panel_features.csv")
OUTPUT_DAILY_STATES_PATH = Path("data/processed/hmm_daily_states.csv")
OUTPUT_SUMMARY_PATH = Path("models/hmm_risk_state_summary.json")

N_STATES = 3
RANDOM_SEED = 42


def build_daily_hmm_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            total_crashes=("crash_count", "sum"),
            crash_rate=("crash_occurred", "mean"),
            avg_crashes_last_7_days=("crashes_last_7_days", "mean"),
            avg_crashes_last_30_days=("crashes_last_30_days", "mean"),
            temperature_2m_mean=("temperature_2m_mean", "mean"),
            precipitation_sum=("precipitation_sum", "mean"),
            windspeed_10m_max=("windspeed_10m_max", "mean"),
            rain_indicator=("rain_indicator", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily


def label_states_by_risk(daily_states: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    state_risk = (
        daily_states.groupby("hidden_state", as_index=False)
        .agg(
            mean_total_crashes=("total_crashes", "mean"),
            mean_crash_rate=("crash_rate", "mean"),
            mean_lag_30=("avg_crashes_last_30_days", "mean"),
        )
        .sort_values(["mean_crash_rate", "mean_total_crashes"], ascending=True)
        .reset_index(drop=True)
    )

    ordered_states = state_risk["hidden_state"].tolist()

    label_map = {}
    if len(ordered_states) == 3:
        label_map = {
            ordered_states[0]: "Low Risk Regime",
            ordered_states[1]: "Moderate Risk Regime",
            ordered_states[2]: "High Risk Regime",
        }
    else:
        for i, s in enumerate(ordered_states):
            label_map[s] = f"Risk Regime {i}"

    daily_states["state_label"] = daily_states["hidden_state"].map(label_map)
    state_risk["state_label"] = state_risk["hidden_state"].map(label_map)

    return daily_states, {
        "state_order_by_risk": ordered_states,
        "state_label_map": label_map,
        "state_risk_summary": state_risk.to_dict(orient="records"),
    }


def main():
    print("Loading full panel features...")
    df = pd.read_csv(INPUT_PATH)

    print("Building daily HMM dataset...")
    daily = build_daily_hmm_dataset(df)

    feature_cols = [
        "total_crashes",
        "crash_rate",
        "avg_crashes_last_7_days",
        "avg_crashes_last_30_days",
        "temperature_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
        "rain_indicator",
    ]

    print("Daily dataset shape:", daily.shape)
    print("Feature columns:", feature_cols)

    X = daily[feature_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Training Gaussian HMM with {N_STATES} states...")
    hmm = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    hmm.fit(X_scaled)

    hidden_states = hmm.predict(X_scaled)
    state_probs = hmm.predict_proba(X_scaled)

    daily["hidden_state"] = hidden_states
    daily["state_probability_max"] = state_probs.max(axis=1)

    daily, state_label_info = label_states_by_risk(daily)

    OUTPUT_DAILY_STATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    daily.to_csv(OUTPUT_DAILY_STATES_PATH, index=False)

    transition_matrix = hmm.transmat_
    start_probabilities = hmm.startprob_

    state_counts = (
        daily["state_label"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    state_means = (
        daily.groupby("state_label", as_index=False)[feature_cols]
        .mean()
        .to_dict(orient="records")
    )

    summary = {
        "model": "GaussianHMM",
        "n_states": N_STATES,
        "n_days": int(len(daily)),
        "date_min": str(daily["date"].min().date()),
        "date_max": str(daily["date"].max().date()),
        "feature_columns": feature_cols,
        "state_counts": state_counts,
        "start_probabilities": start_probabilities.tolist(),
        "transition_matrix": transition_matrix.tolist(),
        "state_means": state_means,
        "state_label_info": state_label_info,
    }

    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved daily HMM states to {OUTPUT_DAILY_STATES_PATH}")
    print(f"Saved HMM summary to {OUTPUT_SUMMARY_PATH}")

    print("\nState counts:")
    print(daily["state_label"].value_counts())

    print("\nAverage features by state:")
    print(daily.groupby("state_label")[feature_cols].mean())


if __name__ == "__main__":
    main()