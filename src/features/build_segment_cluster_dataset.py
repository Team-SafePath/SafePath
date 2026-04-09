from pathlib import Path

import pandas as pd


FULL_PANEL_PATH = Path("data/processed/full_panel_features_with_infra.csv")
INFRA_PATH = Path("data/processed/segment_infrastructure_features.csv")
OUT_PATH = Path("data/processed/segment_cluster_features_with_infra.csv")


def impute_by_road_type(df: pd.DataFrame, value_col: str, road_col: str = "road_type"):
    df = df.copy()

    group_medians = df.groupby(road_col)[value_col].median()
    global_median = df[value_col].median()

    df[value_col] = df.apply(
        lambda r: (
            group_medians.get(r[road_col], global_median)
            if pd.isna(r[value_col])
            else r[value_col]
        ),
        axis=1,
    )

    df[value_col] = df[value_col].fillna(global_median)
    return df


def main():
    print("Loading full panel...")
    panel = pd.read_csv(FULL_PANEL_PATH)

    print("Loading infrastructure features...")
    infra = pd.read_csv(INFRA_PATH)

    print("\nFull panel shape:", panel.shape)
    print("Infra shape:", infra.shape)

    required_panel_cols = {
        "segment_id",
        "crash_count",
        "crash_occurred",
        "crashes_last_7_days",
        "crashes_last_30_days",
        "segment_length",
    }
    missing_panel = required_panel_cols - set(panel.columns)
    if missing_panel:
        raise ValueError(f"Missing required columns in full panel: {missing_panel}")

    if "segment_id" not in infra.columns:
        raise ValueError("segment_infrastructure_features.csv must contain segment_id")

    print("\nAggregating full panel to segment-level features...")
    seg = (
        panel.groupby("segment_id")
        .agg(
            total_crashes=("crash_count", "sum"),
            avg_crash_rate=("crash_occurred", "mean"),
            crash_volatility=("crash_occurred", "std"),
            pct_days_with_crash=("crash_occurred", "mean"),
            avg_crashes_last_7_days=("crashes_last_7_days", "mean"),
            avg_crashes_last_30_days=("crashes_last_30_days", "mean"),
            segment_length=("segment_length", "first"),
        )
        .reset_index()
    )

    seg["crash_volatility"] = seg["crash_volatility"].fillna(0.0)

    # Keep infrastructure columns we want
    infra_keep = [
        "segment_id",
        "road_type",
        "lanes",
        "maxspeed",
        "oneway",
        "is_bridge",
        "is_tunnel",
        "is_junction",
        "segment_curvature",
        "bearing_change_max",
        "u_degree",
        "v_degree",
        "intersection_degree_max",
        "near_intersection",
        "near_traffic_signal",
        "near_crossing",
        "curvature_norm",
        "bearing_norm",
        "intersection_norm",
        "visibility_risk_score",
    ]
    infra_keep = [c for c in infra_keep if c in infra.columns]
    infra = infra[infra_keep].copy()

    merged = seg.merge(infra, on="segment_id", how="left", validate="one_to_one")

    print("\nMerged shape:", merged.shape)

    if "road_type" not in merged.columns:
        merged["road_type"] = "unknown"
    else:
        merged["road_type"] = merged["road_type"].fillna("unknown")

    for col in ["lanes", "maxspeed"]:
        if col in merged.columns:
            merged = impute_by_road_type(merged, col, "road_type")

    numeric_cols = merged.select_dtypes(include=["number", "bool"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "segment_id"]

    for col in numeric_cols:
        if merged[col].isna().any():
            if col in {
                "oneway",
                "is_bridge",
                "is_tunnel",
                "is_junction",
                "near_intersection",
                "near_traffic_signal",
                "near_crossing",
            }:
                merged[col] = merged[col].fillna(0)
            else:
                merged[col] = merged[col].fillna(merged[col].median())

    for col in [
        "oneway",
        "is_bridge",
        "is_tunnel",
        "is_junction",
        "near_intersection",
        "near_traffic_signal",
        "near_crossing",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].round().astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print(f"\nSaved merged segment cluster dataset to {OUT_PATH}")

    print("\nColumns:")
    print(merged.columns.tolist())

    print("\nHead:")
    print(merged.head(10))

    print("\nNull counts:")
    print(merged.isna().sum().sort_values(ascending=False).head(20))

    print("\nNumeric summary (selected):")
    selected = [
        c
        for c in [
            "total_crashes",
            "avg_crash_rate",
            "crash_volatility",
            "pct_days_with_crash",
            "avg_crashes_last_7_days",
            "avg_crashes_last_30_days",
            "segment_length",
            "lanes",
            "maxspeed",
            "segment_curvature",
            "bearing_change_max",
            "intersection_degree_max",
            "visibility_risk_score",
        ]
        if c in merged.columns
    ]
    print(merged[selected].describe())

    print("\nRoad type counts:")
    print(merged["road_type"].value_counts().head(20))


if __name__ == "__main__":
    main()