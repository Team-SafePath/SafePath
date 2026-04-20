from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


NETWORK_PATH = Path("data/raw/nyc_street_network.geojson")
PANEL_PATH = Path("data/processed/full_panel_features_with_infra.csv")
CLUSTERS_PATH = Path("data/processed/segment_clusters_with_infra.csv")
PREDICTIONS_PATH = Path("data/processed/segment_prediction_summary_with_infra.csv")
OUT_PATH = Path("data/processed/segment_combined_map.geojson")

SIMPLIFY_TOLERANCE_METERS = 6.0


def normalize_road_type(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "unknown"
    text = str(val).strip()
    if not text:
        return "unknown"
    return text


def resolve_prefer_y(df: pd.DataFrame, base_name: str):
    x_col = f"{base_name}_x"
    y_col = f"{base_name}_y"

    if base_name in df.columns:
        return df

    if x_col in df.columns and y_col in df.columns:
        df[base_name] = df[y_col].combine_first(df[x_col])
        df = df.drop(columns=[x_col, y_col])
    elif y_col in df.columns:
        df[base_name] = df[y_col]
        df = df.drop(columns=[y_col])
    elif x_col in df.columns:
        df[base_name] = df[x_col]
        df = df.drop(columns=[x_col])

    return df


def main():
    print("Loading street network...")
    edges = gpd.read_file(NETWORK_PATH).reset_index(drop=True)
    edges["segment_id"] = edges.index.astype(int)

    print("Loading enriched full panel...")
    panel = pd.read_csv(PANEL_PATH)

    print("Loading GMM cluster assignments...")
    clusters = pd.read_csv(CLUSTERS_PATH)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_PATH}. "
            "You need a true segment-level prediction summary before exporting the dashboard map."
        )

    print("Loading segment prediction summary...")
    preds = pd.read_csv(PREDICTIONS_PATH)

    required_panel_cols = {
        "segment_id",
        "segment_length",
        "lanes",
        "maxspeed",
        "segment_curvature",
        "bearing_change_max",
        "intersection_degree_max",
        "near_intersection",
        "near_traffic_signal",
        "visibility_risk_score",
        "crash_count",
    }
    missing_panel = required_panel_cols - set(panel.columns)
    if missing_panel:
        raise ValueError(f"Missing required columns in panel: {missing_panel}")

    required_cluster_cols = {"segment_id", "gmm_cluster", "cluster_label"}
    missing_cluster = required_cluster_cols - set(clusters.columns)
    if missing_cluster:
        raise ValueError(f"Missing required columns in clusters: {missing_cluster}")

    required_pred_cols = {"segment_id", "avg_predicted_risk", "risk_percentile"}
    missing_pred = required_pred_cols - set(preds.columns)
    if missing_pred:
        raise ValueError(f"Missing required columns in predictions file: {missing_pred}")

    print("\nAggregating historical + infrastructure fields to segment level...")
    seg_summary = (
        panel.groupby("segment_id")
        .agg(
            total_crashes=("crash_count", "sum"),
            segment_length=("segment_length", "first"),
            lanes=("lanes", "first"),
            maxspeed=("maxspeed", "first"),
            segment_curvature=("segment_curvature", "first"),
            bearing_change_max=("bearing_change_max", "first"),
            intersection_degree_max=("intersection_degree_max", "first"),
            near_intersection=("near_intersection", "first"),
            near_traffic_signal=("near_traffic_signal", "first"),
            visibility_risk_score=("visibility_risk_score", "first"),
        )
        .reset_index()
    )

    seg_summary["log_total_crashes"] = np.log1p(seg_summary["total_crashes"])

    cluster_keep = (
        clusters[["segment_id", "gmm_cluster", "cluster_label"]]
        .drop_duplicates(subset=["segment_id"])
        .copy()
    )

    pred_keep = (
        preds[["segment_id", "avg_predicted_risk", "risk_percentile"]]
        .drop_duplicates(subset=["segment_id"])
        .copy()
    )

    print("Merging summaries into network...")
    merged = edges.merge(
        seg_summary,
        on="segment_id",
        how="left",
        validate="one_to_one",
        suffixes=("_network", "_panel"),
    )
    merged = merged.merge(
        pred_keep,
        on="segment_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        cluster_keep,
        on="segment_id",
        how="left",
        validate="one_to_one",
    )

    print("\nColumns after merge:")
    print(merged.columns.tolist())

    for col in [
        "segment_length",
        "lanes",
        "maxspeed",
        "segment_curvature",
        "bearing_change_max",
        "intersection_degree_max",
        "near_intersection",
        "near_traffic_signal",
        "visibility_risk_score",
    ]:
        merged = resolve_prefer_y(merged, col)

    fallback_pairs = {
        "segment_length": ("segment_length_network", "segment_length_panel"),
        "lanes": ("lanes_network", "lanes_panel"),
        "maxspeed": ("maxspeed_network", "maxspeed_panel"),
    }

    for base, (x_col, y_col) in fallback_pairs.items():
        if base not in merged.columns:
            if x_col in merged.columns and y_col in merged.columns:
                merged[base] = merged[y_col].combine_first(merged[x_col])
                merged = merged.drop(columns=[x_col, y_col])
            elif y_col in merged.columns:
                merged[base] = merged[y_col]
                merged = merged.drop(columns=[y_col])
            elif x_col in merged.columns:
                merged[base] = merged[x_col]
                merged = merged.drop(columns=[x_col])

    if "highway" in merged.columns:
        merged["road_type"] = merged["highway"].apply(normalize_road_type)
    else:
        merged["road_type"] = "unknown"

    # Add exact top-k flags for the dashboard
    merged["is_top1_risk"] = (merged["risk_percentile"] >= 0.99).astype(int)
    merged["is_top5_risk"] = (merged["risk_percentile"] >= 0.95).astype(int)
    merged["is_top10_risk"] = (merged["risk_percentile"] >= 0.90).astype(int)

    keep_cols = [
        "segment_id",
        "road_type",
        "total_crashes",
        "log_total_crashes",
        "avg_predicted_risk",
        "risk_percentile",
        "is_top1_risk",
        "is_top5_risk",
        "is_top10_risk",
        "segment_length",
        "lanes",
        "maxspeed",
        "segment_curvature",
        "bearing_change_max",
        "intersection_degree_max",
        "near_intersection",
        "near_traffic_signal",
        "visibility_risk_score",
        "gmm_cluster",
        "cluster_label",
        "geometry",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged = merged[keep_cols].copy()

    numeric_fill_zero = [
        "total_crashes",
        "log_total_crashes",
        "avg_predicted_risk",
        "risk_percentile",
        "is_top1_risk",
        "is_top5_risk",
        "is_top10_risk",
        "segment_length",
        "lanes",
        "maxspeed",
        "segment_curvature",
        "bearing_change_max",
        "intersection_degree_max",
        "near_intersection",
        "near_traffic_signal",
        "visibility_risk_score",
        "gmm_cluster",
    ]
    for col in numeric_fill_zero:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    merged["road_type"] = merged["road_type"].fillna("unknown")
    merged["cluster_label"] = merged["cluster_label"].fillna("Unassigned")

    for col in [
        "near_intersection",
        "near_traffic_signal",
        "is_top1_risk",
        "is_top5_risk",
        "is_top10_risk",
        "gmm_cluster",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].round().astype(int)

    print(f"Simplifying geometry with tolerance {SIMPLIFY_TOLERANCE_METERS} meters...")
    original_crs = merged.crs
    merged = merged.to_crs(epsg=3857)
    merged["geometry"] = merged["geometry"].simplify(
        SIMPLIFY_TOLERANCE_METERS,
        preserve_topology=True,
    )
    merged = merged.to_crs(original_crs if original_crs is not None else 4326)

    merged = merged[~merged.geometry.is_empty & merged.geometry.notna()].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(OUT_PATH, driver="GeoJSON")

    print(f"\nSaved dashboard GeoJSON to {OUT_PATH}")
    print("\nColumns:")
    print(merged.columns.tolist())

    print("\nHead:")
    print(merged.head(10))

    try:
        size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
        print(f"\nOutput file size: {size_mb:.2f} MB")
    except Exception:
        pass


if __name__ == "__main__":
    main()