from pathlib import Path

import geopandas as gpd
import pandas as pd


CRASHES_PATH = Path("data/interim/nyc_crashes_clean.geojson")
STREETS_PATH = Path("data/raw/nyc_street_network.geojson")
OUTPUT_PATH = Path("data/interim/crashes_with_segments.geojson")

# NYC projected CRS for distance calculations
PROJECTED_CRS = "EPSG:2263"  # NAD83 / New York Long Island (US ft)


def map_crashes_to_segments(max_distance: float = 100):
    """
    Spatially join each crash point to exactly one nearest street segment.

    Parameters
    ----------
    max_distance : float
        Maximum matching distance in feet under EPSG:2263.

    Output
    ------
    data/interim/crashes_with_segments.geojson
        One row per matched crash with a single assigned segment_id.
    """
    print("Loading cleaned crash points...")
    crashes = gpd.read_file(CRASHES_PATH)

    print("Loading street segments...")
    streets = gpd.read_file(STREETS_PATH)

    print(f"Initial crash rows: {len(crashes):,}")
    print(f"Street segment rows: {len(streets):,}")

    # Create a unique crash identifier if one is not already present
    if "collision_id" in crashes.columns:
        missing_collision_id = crashes["collision_id"].isna().sum()
        print(f"Missing collision_id values: {missing_collision_id:,}")

        # Some rows may not have collision_id; fall back to row index for those only
        crashes = crashes.reset_index(drop=True)
        crashes["crash_uid"] = crashes["collision_id"].astype("string")
        missing_mask = crashes["crash_uid"].isna()
        crashes.loc[missing_mask, "crash_uid"] = (
            "missing_collision_" + crashes.index.astype(str)
        )[missing_mask]
    else:
        crashes = crashes.reset_index(drop=True)
        crashes["crash_uid"] = "crash_" + crashes.index.astype(str)

    # Keep useful street columns only
    keep_cols = [
        "u",
        "v",
        "key",
        "length",
        "highway",
        "name",
        "geometry",
    ]
    existing_cols = [col for col in keep_cols if col in streets.columns]
    streets = streets[existing_cols].copy()

    # Stable segment_id
    streets = streets.reset_index(drop=True)
    streets["segment_id"] = streets.index.astype(int)

    print("Projecting datasets...")
    crashes = crashes.to_crs(PROJECTED_CRS)
    streets = streets.to_crs(PROJECTED_CRS)

    print("Running nearest spatial join...")
    joined = gpd.sjoin_nearest(
        crashes,
        streets,
        how="left",
        max_distance=max_distance,
        distance_col="distance_to_segment",
    )

    print(f"Rows after spatial join: {len(joined):,}")

    # Drop unmatched crashes
    unmatched = joined["segment_id"].isna().sum()
    if unmatched > 0:
        print(f"Unmatched crashes to drop: {unmatched:,}")
    joined = joined[joined["segment_id"].notna()].copy()

    # Deduplicate so each crash maps to exactly one segment
    # Sort so the closest segment is kept first, then deterministic tie-break by segment_id
    joined = joined.sort_values(
        by=["crash_uid", "distance_to_segment", "segment_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    before_dedup = len(joined)
    joined = joined.drop_duplicates(subset=["crash_uid"], keep="first").copy()
    removed_dupes = before_dedup - len(joined)

    print(f"Duplicate crash-segment matches removed: {removed_dupes:,}")
    print(f"Final matched crash rows: {len(joined):,}")

    # Type cleanup
    if "segment_id" in joined.columns:
        joined["segment_id"] = joined["segment_id"].astype(int)

    # Optional diagnostics
    if "collision_id" in joined.columns:
        unique_collision_ids = joined["collision_id"].nunique(dropna=True)
        print(f"Unique non-null collision_id values kept: {unique_collision_ids:,}")

    print("Saving output...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joined.to_file(OUTPUT_PATH, driver="GeoJSON")

    print(f"Saved one-to-one crash-segment mapping to {OUTPUT_PATH}")

    return joined


if __name__ == "__main__":
    map_crashes_to_segments()