from pathlib import Path

import numpy as np
import pandas as pd


FULL_PANEL_PATH = Path("data/processed/full_panel_features.csv")
INFRA_PATH = Path("data/processed/segment_infrastructure_features.csv")
OUT_PATH = Path("data/processed/full_panel_features_with_infra.csv")


def impute_by_road_type(df: pd.DataFrame, value_col: str, road_col: str = "road_type"):
    df = df.copy()

    group_medians = df.groupby(road_col)[value_col].median()
    global_median = df[value_col].median()

    df[value_col] = df.apply(
        lambda r: (
            group_medians.get(r[road_col], np.nan)
            if pd.isna(r[value_col])
            else r[value_col]
        ),
        axis=1,
    )

    df[value_col] = df[value_col].fillna(global_median)

    return df


def main():
    print("Loading existing full panel...")
    panel = pd.read_csv(FULL_PANEL_PATH)

    print("Loading infrastructure features...")
    infra = pd.read_csv(INFRA_PATH)

    print("\nFull panel shape:", panel.shape)
    print("Infrastructure shape:", infra.shape)

    required_panel_cols = {"segment_id"}
    required_infra_cols = {
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
        "visibility_risk_score",
    }

    missing_panel = required_panel_cols - set(panel.columns)
    missing_infra = required_infra_cols - set(infra.columns)

    if missing_panel:
        raise ValueError(f"Missing required columns in full panel: {missing_panel}")
    if missing_infra:
        raise ValueError(f"Missing required columns in infrastructure file: {missing_infra}")

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

    numeric_cols = [
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
    numeric_cols = [c for c in numeric_cols if c in infra.columns]

    for col in numeric_cols:
        infra[col] = pd.to_numeric(infra[col], errors="coerce")

    print("\nNull counts before imputation:")
    print(infra[numeric_cols].isna().sum().sort_values(ascending=False))

    print("\nImputing lanes and maxspeed by road_type median...")
    if "lanes" in infra.columns:
        infra = impute_by_road_type(infra, "lanes", "road_type")
    if "maxspeed" in infra.columns:
        infra = impute_by_road_type(infra, "maxspeed", "road_type")

    fill_zero_cols = [
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
    fill_zero_cols = [c for c in fill_zero_cols if c in infra.columns]

    for col in fill_zero_cols:
        infra[col] = infra[col].fillna(0)

    indicator_cols = [
        "oneway",
        "is_bridge",
        "is_tunnel",
        "is_junction",
        "near_intersection",
        "near_traffic_signal",
        "near_crossing",
    ]
    indicator_cols = [c for c in indicator_cols if c in infra.columns]

    for col in indicator_cols:
        infra[col] = infra[col].round().astype(int)

    print("\nNull counts after imputation:")
    print(infra[numeric_cols].isna().sum().sort_values(ascending=False))

    merge_cols = [
        "segment_id",
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
    merge_cols = [c for c in merge_cols if c in infra.columns]

    infra_merge = infra[merge_cols].copy()

    print("\nMerging infrastructure into full panel...")
    enriched = panel.merge(
        infra_merge,
        on="segment_id",
        how="left",
        validate="many_to_one",
    )

    print("Merged shape:", enriched.shape)

    new_feature_cols = [c for c in merge_cols if c != "segment_id"]
    print("\nNull counts in merged new columns:")
    print(enriched[new_feature_cols].isna().sum().sort_values(ascending=False))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUT_PATH, index=False)

    print(f"\nSaved enriched full panel to {OUT_PATH}")
    print("\nNew columns added:")
    print(new_feature_cols)

    print("\nHead of new columns:")
    print(enriched[["segment_id"] + new_feature_cols].head(10))


if __name__ == "__main__":
    main()