from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np


SEGMENTS_PATH = Path("data/raw/nyc_street_network.geojson")
POSITIVE_PATH = Path("data/processed/segment_daily_crashes.csv")
OUTPUT_PATH = Path("data/processed/modeling_dataset_sampled.csv")

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
NEGATIVE_RATIO = 1.0
RANDOM_SEED = 42


def sample_negative_examples(
    negative_ratio: float = NEGATIVE_RATIO,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build a sampled modeling dataset with:
      - positive rows: observed segment-day crash events
      - negative rows: sampled segment-day combinations with no crash

    Output columns:
      segment_id, date, crash_count, crash_occurred
    """
    print("Loading positive crash examples...")
    positives = pd.read_csv(POSITIVE_PATH)
    positives["date"] = pd.to_datetime(positives["date"])
    positives["segment_id"] = positives["segment_id"].astype(str)

    print(f"Positive rows: {len(positives):,}")

    print("Loading street segments...")
    segments = gpd.read_file(SEGMENTS_PATH).reset_index(drop=True)
    segments["segment_id"] = segments.index.astype(str)

    segment_ids = segments["segment_id"].to_numpy()
    print(f"Available segments: {len(segment_ids):,}")

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    print(f"Available dates: {len(dates):,}")

    n_positives = len(positives)
    n_negatives = int(n_positives * negative_ratio)

    print(f"Target negative samples: {n_negatives:,}")

    positive_pairs = set(
        zip(
            positives["segment_id"],
            positives["date"].dt.strftime("%Y-%m-%d"),
        )
    )

    rng = np.random.default_rng(random_seed)

    negatives = []
    seen_negative_pairs = set()

    print("Sampling negative examples...")

    while len(negatives) < n_negatives:
        remaining = n_negatives - len(negatives)
        batch_size = min(max(remaining * 2, 10000), 250000)

        sampled_segments = rng.choice(segment_ids, size=batch_size, replace=True)
        sampled_dates = rng.choice(dates, size=batch_size, replace=True)

        for seg, dt in zip(sampled_segments, sampled_dates):
            dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
            pair = (str(seg), dt_str)

            if pair in positive_pairs:
                continue

            if pair in seen_negative_pairs:
                continue

            seen_negative_pairs.add(pair)
            negatives.append((str(seg), dt_str, 0))

            if len(negatives) >= n_negatives:
                break

        print(f"Collected negatives: {len(negatives):,} / {n_negatives:,}")

    negatives_df = pd.DataFrame(
        negatives,
        columns=["segment_id", "date", "crash_count"]
    )
    negatives_df["date"] = pd.to_datetime(negatives_df["date"])

    positives_out = positives.copy()
    positives_out["crash_count"] = positives_out["crash_count"].astype(int)

    modeling_df = pd.concat(
        [positives_out[["segment_id", "date", "crash_count"]], negatives_df],
        ignore_index=True
    )

    modeling_df["crash_occurred"] = (modeling_df["crash_count"] > 0).astype(int)

    modeling_df = modeling_df.sort_values(
        ["date", "segment_id"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    modeling_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved modeling dataset to {OUTPUT_PATH}")
    print(f"Total rows: {len(modeling_df):,}")
    print("\nClass balance:")
    print(modeling_df["crash_occurred"].value_counts())

    return modeling_df


if __name__ == "__main__":
    sample_negative_examples()