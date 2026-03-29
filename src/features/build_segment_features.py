from pathlib import Path
import geopandas as gpd
import pandas as pd


STREETS_PATH = Path("data/raw/nyc_street_network.geojson")
OUTPUT_PATH = Path("data/processed/segment_features.csv")


def normalize_road_type(val):
    # Handle missing
    if val is None:
        return "unknown"

    # If it's a list / array → take first element
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) > 0 else "unknown"

    # Handle numpy arrays
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return val[0] if len(val) > 0 else "unknown"
    except:
        pass

    # Convert to string
    val = str(val)

    # Handle nan string cases
    if val.lower() == "nan":
        return "unknown"

    # If multiple categories → take first
    if " " in val:
        return val.split(" ")[0]

    return val

def build_segment_features():
    print("Loading street network...")
    gdf = gpd.read_file(STREETS_PATH)

    print("Building infrastructure features...")

    gdf = gdf.reset_index(drop=True)
    gdf["segment_id"] = gdf.index.astype(int)

    df = gdf[["segment_id", "length", "highway"]].copy()

    df = df.rename(columns={
        "length": "segment_length",
        "highway": "road_type"
    })

    # ✅ FIX HERE
    df["road_type"] = df["road_type"].apply(normalize_road_type)

    print("\nCleaned road_type examples:")
    print(df["road_type"].value_counts().head(10))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved segment features to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    build_segment_features()