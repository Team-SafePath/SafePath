from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


NETWORK_PATH = Path("data/raw/nyc_street_network.geojson")
PANEL_PATH = Path("data/processed/full_panel_features_with_infra.csv")
CLUSTERS_PATH = Path("data/processed/segment_clusters_with_infra.csv")
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
    """
    Resolve columns like lanes_x / lanes_y into a single lanes column,
    preferring the merged-in summary value (_y) when present, then falling
    back to the original network value (_x).
    """
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
        "crash_occurred",
    }
    missing_panel = required_panel_cols - set(panel.columns)
    if missing_panel:
        raise ValueError(f"Missing required columns in panel: {missing_panel}")

    required_cluster_cols = {"segment_id", "gmm_cluster", "cluster_label"}
    missing_cluster = required_cluster_cols - set(clusters.columns)
    if missing_cluster:
        raise ValueError(f"Missing required columns in clusters: {missing_cluster}")

    print("\nAggregating panel to segment-level dashboard fields...")
    seg_summary = (
        panel.groupby("segment_id")
        .agg(
            total_crashes=("crash_count", "sum"),
            avg_predicted_risk=("crash_occurred", "mean"),
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

    max_risk = seg_summary["avg_predicted_risk"].max()
    if pd.notna(max_risk) and max_risk > 0:
        seg_summary["normalized_risk"] = seg_summary["avg_predicted_risk"] / max_risk
    else:
        seg_summary["normalized_risk"] = 0.0

    cluster_keep = (
        clusters[["segment_id", "gmm_cluster", "cluster_label"]]
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
        cluster_keep,
        on="segment_id",
        how="left",
        validate="one_to_one",
    )

    print("\nColumns after merge:")
    print(merged.columns.tolist())

    # Resolve potential duplicate columns introduced by merge
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

    # Also handle the explicit suffix names from the merge above if needed
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

    # Resolve road type from network
    if "highway" in merged.columns:
        merged["road_type"] = merged["highway"].apply(normalize_road_type)
    else:
        merged["road_type"] = "unknown"

    keep_cols = [
        "segment_id",
        "road_type",
        "total_crashes",
        "avg_predicted_risk",
        "normalized_risk",
        "log_total_crashes",
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

    missing_keep = {
        "segment_id",
        "road_type",
        "total_crashes",
        "avg_predicted_risk",
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
    } - set(keep_cols)
    if missing_keep:
        print("\nWarning: these expected dashboard fields are still missing:")
        print(sorted(missing_keep))

    merged = merged[keep_cols].copy()

    numeric_fill_zero = [
        "total_crashes",
        "avg_predicted_risk",
        "normalized_risk",
        "log_total_crashes",
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

    if "road_type" in merged.columns:
        merged["road_type"] = merged["road_type"].fillna("unknown")
    if "cluster_label" in merged.columns:
        merged["cluster_label"] = merged["cluster_label"].fillna("Unassigned")

    if "near_intersection" in merged.columns:
        merged["near_intersection"] = merged["near_intersection"].round().astype(int)
    if "near_traffic_signal" in merged.columns:
        merged["near_traffic_signal"] = merged["near_traffic_signal"].round().astype(int)
    if "gmm_cluster" in merged.columns:
        merged["gmm_cluster"] = merged["gmm_cluster"].round().astype(int)

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