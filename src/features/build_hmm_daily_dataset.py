from pathlib import Path

import pandas as pd


FULL_PANEL_PATH = Path("data/processed/full_panel_features_with_infra.csv")
OUT_PATH = Path("data/processed/hmm_daily_features_with_infra.csv")


def main():
    print("Loading enriched full panel...")
    df = pd.read_csv(FULL_PANEL_PATH)

    required_cols = {
        "date",
        "crash_count",
        "crash_occurred",
        "crashes_last_7_days",
        "crashes_last_30_days",
        "temperature_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
        "rain_indicator",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in full panel: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    print("\nAggregating to daily dataset...")
    daily = (
        df.groupby("date")
        .agg(
            total_crashes=("crash_count", "sum"),
            total_segments=("segment_id", "nunique"),
            total_positive_segments=("crash_occurred", "sum"),
            crash_rate=("crash_occurred", "mean"),
            avg_crashes_last_7_days=("crashes_last_7_days", "mean"),
            avg_crashes_last_30_days=("crashes_last_30_days", "mean"),
            temperature_2m_mean=("temperature_2m_mean", "mean"),
            precipitation_sum=("precipitation_sum", "mean"),
            windspeed_10m_max=("windspeed_10m_max", "mean"),
            rain_indicator=("rain_indicator", "mean"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Rolling context
    daily["crashes_7d_avg_city"] = (
        daily["total_crashes"].rolling(window=7, min_periods=1).mean()
    )
    daily["crashes_30d_avg_city"] = (
        daily["total_crashes"].rolling(window=30, min_periods=1).mean()
    )
    daily["crash_rate_7d_avg_city"] = (
        daily["crash_rate"].rolling(window=7, min_periods=1).mean()
    )
    daily["crash_rate_30d_avg_city"] = (
        daily["crash_rate"].rolling(window=30, min_periods=1).mean()
    )

    # Calendar fields
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_PATH, index=False)

    print(f"\nSaved daily HMM dataset to {OUT_PATH}")
    print("\nShape:")
    print(daily.shape)

    print("\nColumns:")
    print(daily.columns.tolist())

    print("\nHead:")
    print(daily.head(10))

    print("\nNull counts:")
    print(daily.isna().sum().sort_values(ascending=False).head(20))

    print("\nSummary:")
    print(
        daily[
            [
                "total_crashes",
                "crash_rate",
                "avg_crashes_last_7_days",
                "avg_crashes_last_30_days",
                "temperature_2m_mean",
                "precipitation_sum",
                "windspeed_10m_max",
                "crashes_7d_avg_city",
                "crashes_30d_avg_city",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()