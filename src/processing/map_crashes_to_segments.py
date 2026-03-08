from pathlib import Path

import geopandas as gpd
import pandas as pd


CRASHES_PATH = Path("data/interim/nyc_crashes_clean.geojson")
STREETS_PATH = Path("data/raw/nyc_street_network.geojson")
OUTPUT_PATH = Path("data/interim/crashes_with_segments.geojson")

# Use a projected CRS for NYC so distance calculations work in meters
PROJECTED_CRS = "EPSG:2263"  # NAD83 / New York Long Island (ftUS)


def map_crashes_to_segments(max_distance: float = 100):
    """
    Spatially join each crash point to its nearest street segment.

    Parameters
    ----------
    max_distance : float
        Maximum allowed distance for matching a crash to a street segment.
        Since EPSG:2263 uses US survey feet, 100 = ~100 feet.

    Returns
    -------
    geopandas.GeoDataFrame
        Crash records with assigned street segment information.
    """
    print("Loading cleaned crash points...")
    crashes = gpd.read_file(CRASHES_PATH)

    print("Loading street segments...")
    streets = gpd.read_file(STREETS_PATH)

    print(f"Crash rows: {len(crashes):,}")
    print(f"Street segment rows: {len(streets):,}")

    # Keep only needed columns from streets
    street_cols = [col for col in ["u", "v", "key", "length", "highway", "name", "geometry"] if col in streets.columns]
    streets = streets[street_cols].copy()

    # Create a stable segment ID
    streets = streets.reset_index(drop=True)
    streets["segment_id"] = streets.index.astype(str)

    print("Projecting both datasets to NYC projected CRS...")
    crashes = crashes.to_crs(PROJECTED_CRS)
    streets = streets.to_crs(PROJECTED_CRS)

    print("Running nearest spatial join...")
    crashes_with_segments = gpd.sjoin_nearest(
        crashes,
        streets,
        how="left",
        max_distance=max_distance,
        distance_col="distance_to_segment"
    )

    matched = crashes_with_segments["segment_id"].notna().sum()
    unmatched = len(crashes_with_segments) - matched

    print(f"Matched crashes: {matched:,}")
    print(f"Unmatched crashes: {unmatched:,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Saving output...")
    crashes_with_segments.to_file(OUTPUT_PATH, driver="GeoJSON")

    print(f"Saved crash-segment mapping to {OUTPUT_PATH}")

    return crashes_with_segments


if __name__ == "__main__":
    map_crashes_to_segments()