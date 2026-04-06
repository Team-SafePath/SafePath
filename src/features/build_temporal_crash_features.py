from pathlib import Path
import numpy as np
import pandas as pd


MODEL_INPUT_PATH = Path("data/processed/modeling_dataset_full.csv")
CRASH_HISTORY_PATH = Path("data/processed/segment_daily_crashes.csv")
OUTPUT_PATH = Path("data/processed/modeling_dataset_final.csv")


def build_temporal_crash_features() -> pd.DataFrame:
    """
    Add temporal crash-history and cyclical calendar features.

    Inputs
    ------
    data/processed/modeling_dataset_full.csv
        Sampled modeling dataset with date still present.

    data/processed/segment_daily_crashes.csv
        Full positive crash history by segment and day.

    Output
    ------
    data/processed/modeling_dataset_final.csv
    """
    print("Loading modeling dataset...")
    model_df = pd.read_csv(MODEL_INPUT_PATH)

    print("Loading full crash history...")
    history_df = pd.read_csv(CRASH_HISTORY_PATH)

    model_df["date"] = pd.to_datetime(model_df["date"], errors="coerce")
    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")

    model_df["segment_id"] = model_df["segment_id"].astype(str)
    history_df["segment_id"] = history_df["segment_id"].astype(str)

    history_df["crash_count"] = history_df["crash_count"].fillna(0).astype(int)

    print("Building lagged crash features from historical crash counts...")

    lag_features = []

    for segment_id, group in history_df.groupby("segment_id", sort=False):
        group = group.sort_values("date").copy()

        # Shift by 1 so current-day crashes are not included
        shifted = group["crash_count"].shift(1)

        group["crashes_last_7_days"] = shifted.rolling(
            window=7, min_periods=1
        ).sum()

        group["crashes_last_30_days"] = shifted.rolling(
            window=30, min_periods=1
        ).sum()

        lag_features.append(
            group[
                [
                    "segment_id",
                    "date",
                    "crashes_last_7_days",
                    "crashes_last_30_days",
                ]
            ]
        )

    lag_df = pd.concat(lag_features, ignore_index=True)

    print("Merging lagged crash features into modeling dataset...")
    df = model_df.merge(
        lag_df,
        on=["segment_id", "date"],
        how="left",
    )

    df["crashes_last_7_days"] = df["crashes_last_7_days"].fillna(0)
    df["crashes_last_30_days"] = df["crashes_last_30_days"].fillna(0)

    print("Adding cyclical calendar features...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved final feature dataset to {OUTPUT_PATH}")
    print("\nNew columns added:")
    print(
        [
            "crashes_last_7_days",
            "crashes_last_30_days",
            "sin_month",
            "cos_month",
            "sin_day_of_week",
            "cos_day_of_week",
        ]
    )
    print("\nPreview:")
    print(df.head())

    return df


if __name__ == "__main__":
    build_temporal_crash_features()