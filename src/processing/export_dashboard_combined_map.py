from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np


STREETS_PATH = Path("data/raw/nyc_street_network.geojson")
CRASHES_PATH = Path("data/processed/segment_daily_crashes.csv")
PREDICTIONS_PATH = Path("data/processed/full_panel_predictions.csv")
SEGMENT_FEATURES_PATH = Path("data/processed/segment_features.csv")

OUTPUT_PATH = Path("dashboard_exports/segment_combined_map.geojson")


def main():
    print("Loading street network...")
    streets = gpd.read_file(STREETS_PATH)
    streets = streets.reset_index(drop=True).copy()
    streets["segment_id"] = streets.index.astype(int)

    print("Loading crash totals...")
    crashes = pd.read_csv(CRASHES_PATH)
    crash_totals = (
        crashes.groupby("segment_id", as_index=False)["crash_count"]
        .sum()
        .rename(columns={"crash_count": "total_crashes"})
    )

    print("Loading predictions...")
    preds = pd.read_csv(PREDICTIONS_PATH)

    required_pred_cols = {"segment_id", "predicted_proba"}
    missing = required_pred_cols - set(preds.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    risk_by_segment = (
        preds.groupby("segment_id", as_index=False)["predicted_proba"]
        .mean()
        .rename(columns={"predicted_proba": "avg_predicted_risk"})
    )

    print("Loading segment features...")
    features = pd.read_csv(SEGMENT_FEATURES_PATH)[
        ["segment_id", "segment_length", "road_type"]
    ].copy()

    print("Merging...")
    gdf = streets.merge(crash_totals, on="segment_id", how="left")
    gdf = gdf.merge(risk_by_segment, on="segment_id", how="left")
    gdf = gdf.merge(features, on="segment_id", how="left")

    gdf["total_crashes"] = gdf["total_crashes"].fillna(0).astype(int)
    gdf["avg_predicted_risk"] = gdf["avg_predicted_risk"].fillna(0.0)
    gdf["segment_length"] = gdf["segment_length"].fillna(0)

    # Log helpers for map coloring
    gdf["log_total_crashes"] = np.log1p(gdf["total_crashes"])

    if gdf["avg_predicted_risk"].max() > 0:
        gdf["normalized_risk"] = gdf["avg_predicted_risk"] / gdf["avg_predicted_risk"].max()
    else:
        gdf["normalized_risk"] = 0.0

    keep_cols = [
        "segment_id",
        "total_crashes",
        "log_total_crashes",
        "avg_predicted_risk",
        "normalized_risk",
        "segment_length",
        "road_type",
        "geometry",
    ]
    gdf = gdf[keep_cols].copy()

    # keep segments that have either crash history or nonzero modeled risk
    gdf = gdf[
        (gdf["total_crashes"] > 0) | (gdf["avg_predicted_risk"] > 0)
    ].copy()

    if gdf.crs is not None and str(gdf.crs).lower() != "epsg:4326":
        gdf = gdf.to_crs(epsg=4326)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT_PATH, driver="GeoJSON")

    print(f"Saved map export to {OUTPUT_PATH}")
    print(f"Segments exported: {len(gdf):,}")
    print(
        gdf[
            ["segment_id", "total_crashes", "avg_predicted_risk", "road_type"]
        ].head()
    )


if __name__ == "__main__":
    main()