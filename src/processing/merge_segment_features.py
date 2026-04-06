from pathlib import Path
import pandas as pd


MODEL_PATH = Path("data/processed/modeling_dataset_with_weather.csv")
SEGMENT_FEATURES_PATH = Path("data/processed/segment_features.csv")

OUTPUT_PATH = Path("data/processed/modeling_dataset_full.csv")


def merge_segment_features():

    print("Loading modeling dataset...")
    df = pd.read_csv(MODEL_PATH)

    print("Loading segment features...")
    segments = pd.read_csv(SEGMENT_FEATURES_PATH)

    df["segment_id"] = df["segment_id"].astype(str)
    segments["segment_id"] = segments["segment_id"].astype(str)

    print("Merging segment infrastructure features...")

    merged = df.merge(
        segments,
        on="segment_id",
        how="left"
    )

    print("Checking for missing segment features...")
    print(merged.isna().sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved final modeling dataset to {OUTPUT_PATH}")

    return merged


if __name__ == "__main__":
    merge_segment_features()