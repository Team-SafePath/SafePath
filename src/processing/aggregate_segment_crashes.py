from pathlib import Path

import pandas as pd
import geopandas as gpd


INPUT_PATH = Path("data/interim/crashes_with_segments.geojson")
OUTPUT_PATH = Path("data/processed/segment_daily_crashes.csv")


def aggregate_segment_crashes() -> pd.DataFrame:
    """
    Aggregate crash records to daily crash counts per street segment.

    Input:
        data/interim/crashes_with_segments.geojson
            One row per crash, with segment_id assigned from spatial join.

    Output:
        data/processed/segment_daily_crashes.csv
            One row per segment_id + date with crash_count.
    """
    print("Loading crash-to-segment mapping...")
    gdf = gpd.read_file(INPUT_PATH)

    print(f"Initial rows: {len(gdf):,}")

    # Keep only crashes successfully matched to a street segment
    df = gdf[gdf["segment_id"].notna()].copy()

    print(f"Rows with matched segment_id: {len(df):,}")

    # Parse date
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    df = df[df["crash_date"].notna()].copy()

    # Convert to daily granularity
    df["date"] = df["crash_date"].dt.floor("D")

    # Aggregate crash counts
    daily_counts = (
        df.groupby(["segment_id", "date"], as_index=False)
        .size()
        .rename(columns={"size": "crash_count"})
        .sort_values(["segment_id", "date"])
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_counts.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved aggregated dataset to {OUTPUT_PATH}")
    print(f"Output rows: {len(daily_counts):,}")
    print("\nPreview:")
    print(daily_counts.head())

    return daily_counts


if __name__ == "__main__":
    aggregate_segment_crashes()