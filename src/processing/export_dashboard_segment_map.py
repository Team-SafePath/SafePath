from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import numpy as np


STREETS_PATH = Path("data/raw/nyc_street_network.geojson")
CRASHES_PATH = Path("data/processed/segment_daily_crashes.csv")
SEGMENT_FEATURES_PATH = Path("data/processed/segment_features.csv")

OUTPUT_PATH = Path("dashboard_exports/segment_crash_map.geojson")


def main():
    print("Loading street network...")
    streets = gpd.read_file(STREETS_PATH)
    streets = streets.reset_index(drop=True).copy()
    streets["segment_id"] = streets.index.astype(int)

    print("Loading crash totals...")
    crashes = pd.read_csv(CRASHES_PATH)
    segment_totals = (
        crashes.groupby("segment_id", as_index=False)["crash_count"]
        .sum()
        .rename(columns={"crash_count": "total_crashes"})
    )

    print("Loading segment features...")
    features = pd.read_csv(SEGMENT_FEATURES_PATH)[
        ["segment_id", "segment_length", "road_type"]
    ].copy()

    print("Merging...")
    gdf = streets.merge(segment_totals, on="segment_id", how="left")
    gdf = gdf.merge(features, on="segment_id", how="left")

    gdf["total_crashes"] = gdf["total_crashes"].fillna(0).astype(int)
    gdf["segment_length"] = gdf["segment_length"].fillna(0)

    # Keep only useful columns
    keep_cols = ["segment_id", "total_crashes", "segment_length", "road_type", "geometry"]
    gdf = gdf[keep_cols].copy()

    # Optional readability filter:
    # only keep segments with at least 1 crash
    gdf = gdf[gdf["total_crashes"] > 0].copy()

    # Reproject for web maps
    if gdf.crs is not None and str(gdf.crs).lower() != "epsg:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Create a log-scaled helper field for better coloring
    gdf["log_total_crashes"] = np.log1p(gdf["total_crashes"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT_PATH, driver="GeoJSON")

    print(f"Saved map export to {OUTPUT_PATH}")
    print(f"Segments exported: {len(gdf):,}")
    print(gdf[["segment_id", "total_crashes", "road_type"]].head())


if __name__ == "__main__":
    main()