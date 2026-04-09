from pathlib import Path
import json

import numpy as np
import pandas as pd


CRASH_HISTORY_PATH = Path("data/processed/segment_daily_crashes.csv")
WEATHER_PATH = Path("data/raw/nyc_weather_2018_2024.csv")
SEGMENT_FEATURES_PATH = Path("data/processed/segment_features.csv")

OUTPUT_PATH = Path("data/processed/lstm/lstm_panel.csv")
SUMMARY_PATH = Path("data/processed/lstm/lstm_panel_summary.json")

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

TRAIN_END_DATE = "2022-12-31"
VALID_END_DATE = "2023-12-31"

MIN_TOTAL_CRASHES_PER_SEGMENT = 20


def normalize_road_type(x):
    if pd.isna(x):
        return "unknown"

    x = str(x).strip()

    if x.startswith("[") and x.endswith("]"):
        x = x[1:-1].replace("'", "").replace('"', "").strip()

    if "," in x:
        x = x.split(",")[0].strip()
    else:
        x = x.split()[0].strip() if x else "unknown"

    return x if x else "unknown"


def load_inputs():
    print("Loading crash history...")
    crashes = pd.read_csv(CRASH_HISTORY_PATH)
    crashes["date"] = pd.to_datetime(crashes["date"], errors="coerce")
    crashes["segment_id"] = crashes["segment_id"].astype(str)
    crashes["crash_count"] = crashes["crash_count"].fillna(0).astype(int)
    crashes = crashes[crashes["date"].notna()].copy()

    print("Loading weather...")
    weather = pd.read_csv(WEATHER_PATH)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather = weather[weather["date"].notna()].copy()

    print("Loading segment features...")
    segments = pd.read_csv(SEGMENT_FEATURES_PATH)
    segments["segment_id"] = segments["segment_id"].astype(str)

    return crashes, weather, segments


def filter_segments(crashes: pd.DataFrame) -> pd.DataFrame:
    print("Filtering segments for LSTM panel...")

    segment_totals = crashes.groupby("segment_id", as_index=True)["crash_count"].sum()
    valid_segments = segment_totals[segment_totals >= MIN_TOTAL_CRASHES_PER_SEGMENT].index

    filtered = crashes[crashes["segment_id"].isin(valid_segments)].copy()

    print(f"Kept segments: {len(valid_segments):,}")
    print(f"Filtered crash rows: {len(filtered):,}")

    return filtered


def build_segment_day_panel(crashes: pd.DataFrame) -> pd.DataFrame:
    print("Building continuous segment-day panel...")

    segment_ids = pd.Index(sorted(crashes["segment_id"].unique()))
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    panel = pd.MultiIndex.from_product(
        [segment_ids, dates],
        names=["segment_id", "date"]
    ).to_frame(index=False)

    daily = crashes.groupby(["segment_id", "date"], as_index=False)["crash_count"].sum()

    panel = panel.merge(
        daily,
        on=["segment_id", "date"],
        how="left"
    )

    panel["crash_count"] = panel["crash_count"].fillna(0).astype(int)
    panel["crash_occurred"] = (panel["crash_count"] > 0).astype(int)

    print(f"Panel rows: {len(panel):,}")
    print(f"Segments in panel: {panel['segment_id'].nunique():,}")

    return panel


def add_weather_features(panel: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    print("Merging weather features...")

    keep_cols = [
        "date",
        "temperature_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
    ]
    weather = weather[[col for col in keep_cols if col in weather.columns]].copy()

    panel = panel.merge(weather, on="date", how="left")

    if "precipitation_sum" in panel.columns and "rain_indicator" not in panel.columns:
        panel["rain_indicator"] = (panel["precipitation_sum"].fillna(0) > 0).astype(int)

    return panel


def add_segment_features(panel: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    print("Merging segment features...")

    segments = segments.copy()

    if "road_type" in segments.columns:
        segments["road_type"] = segments["road_type"].apply(normalize_road_type)
        road_dummies = pd.get_dummies(segments["road_type"], prefix="road", dtype=int)
        segments = pd.concat([segments.drop(columns=["road_type"]), road_dummies], axis=1)

    panel = panel.merge(segments, on="segment_id", how="left")

    return panel


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    print("Adding calendar features...")

    panel["day_of_week"] = panel["date"].dt.dayofweek
    panel["month"] = panel["date"].dt.month

    panel["sin_month"] = np.sin(2 * np.pi * panel["month"] / 12)
    panel["cos_month"] = np.cos(2 * np.pi * panel["month"] / 12)

    panel["sin_day_of_week"] = np.sin(2 * np.pi * panel["day_of_week"] / 7)
    panel["cos_day_of_week"] = np.cos(2 * np.pi * panel["day_of_week"] / 7)

    return panel


def finalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    print("Finalizing panel...")

    bool_cols = panel.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        panel[col] = panel[col].astype(int)

    numeric_cols = panel.select_dtypes(include=["number"]).columns
    panel[numeric_cols] = panel[numeric_cols].fillna(0)

    panel = panel.sort_values(["segment_id", "date"]).reset_index(drop=True)

    return panel


def build_summary(panel: pd.DataFrame) -> dict:
    train_mask = panel["date"] <= pd.Timestamp(TRAIN_END_DATE)
    valid_mask = (
        (panel["date"] > pd.Timestamp(TRAIN_END_DATE))
        & (panel["date"] <= pd.Timestamp(VALID_END_DATE))
    )
    test_mask = panel["date"] > pd.Timestamp(VALID_END_DATE)

    summary = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "min_total_crashes_per_segment": MIN_TOTAL_CRASHES_PER_SEGMENT,
        "rows": int(len(panel)),
        "segments": int(panel["segment_id"].nunique()),
        "columns": list(panel.columns),
        "positive_rate_overall": float(panel["crash_occurred"].mean()),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_positive_rate": float(panel.loc[train_mask, "crash_occurred"].mean()),
        "valid_positive_rate": float(panel.loc[valid_mask, "crash_occurred"].mean()),
        "test_positive_rate": float(panel.loc[test_mask, "crash_occurred"].mean()),
    }

    return summary


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    crashes, weather, segments = load_inputs()
    crashes = filter_segments(crashes)

    panel = build_segment_day_panel(crashes)
    panel = add_weather_features(panel, weather)
    panel = add_segment_features(panel, segments)
    panel = add_calendar_features(panel)
    panel = finalize_panel(panel)

    print(f"Saving panel to {OUTPUT_PATH} ...")
    panel.to_csv(OUTPUT_PATH, index=False)

    summary = build_summary(panel)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {SUMMARY_PATH}")
    print("\nPanel summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()