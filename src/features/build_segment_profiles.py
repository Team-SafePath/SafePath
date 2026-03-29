import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/processed/full_panel_features.csv"
OUTPUT_PATH = "data/processed/segment_profiles.csv"


def build_segment_profiles():
    print("Loading full panel dataset...")
    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    print("Building segment-level aggregates...")

    grouped = df.groupby("segment_id")

    profiles = pd.DataFrame({
        "segment_id": grouped.size().index,

        # Crash behavior
        "total_crashes": grouped["crash_count"].sum().values,
        "avg_crash_rate": grouped["crash_occurred"].mean().values,
        "crash_volatility": grouped["crash_count"].std().fillna(0).values,
        "pct_days_with_crash": grouped["crash_occurred"].mean().values,

        # Lag features
        "avg_crashes_last_7_days": grouped["crashes_last_7_days"].mean().values,
        "avg_crashes_last_30_days": grouped["crashes_last_30_days"].mean().values,

        # Weather sensitivity
        "avg_temp": grouped["temperature_2m_mean"].mean().values,
        "avg_precip": grouped["precipitation_sum"].mean().values,
        "avg_wind": grouped["windspeed_10m_max"].mean().values,

        # Rain vs no rain crash rate
        "crash_rate_rain": grouped.apply(
            lambda g: g[g["rain_indicator"] == 1]["crash_occurred"].mean()
        ).fillna(0).values,

        "crash_rate_no_rain": grouped.apply(
            lambda g: g[g["rain_indicator"] == 0]["crash_occurred"].mean()
        ).fillna(0).values,
    })

    # Infrastructure (take first since constant per segment)
    infra_cols = [
        "segment_length",
        "road_busway", "road_living_street", "road_motorway",
        "road_motorway_link", "road_primary", "road_primary_link",
        "road_residential", "road_secondary", "road_secondary_link",
        "road_tertiary", "road_tertiary_link", "road_trunk",
        "road_trunk_link", "road_unclassified"
    ]

    infra = df.groupby("segment_id")[infra_cols].first().reset_index()

    profiles = profiles.merge(infra, on="segment_id", how="left")

    print("Final profile shape:", profiles.shape)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    profiles.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved segment profiles to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_segment_profiles()