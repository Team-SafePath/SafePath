from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed/segment_cluster_features_with_infra.csv")
OUTPUT_ASSIGNMENTS_PATH = Path("data/processed/segment_clusters_with_infra.csv")
OUTPUT_SUMMARY_PATH = Path("data/processed/gmm_cluster_summary_with_infra.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "gmm_segment_clusters_with_infra.joblib"
SCALER_PATH = MODEL_DIR / "gmm_segment_clusters_with_infra_scaler.joblib"
METRICS_PATH = MODEL_DIR / "gmm_segment_clusters_with_infra_metrics.json"

N_CLUSTERS = 5
RANDOM_STATE = 42


def build_cluster_label_map(summary_df: pd.DataFrame):
    ordered = summary_df.sort_values("avg_crash_rate", ascending=False).reset_index(drop=True)

    labels = {}
    n = len(ordered)

    for i, row in ordered.iterrows():
        cluster_id = int(row["gmm_cluster"])
        if i == 0:
            labels[cluster_id] = "High-Risk Persistent Segments"
        elif i == 1:
            labels[cluster_id] = "Elevated Risk Corridors"
        elif i == 2:
            labels[cluster_id] = "Moderate-Risk Segments"
        elif i == n - 1:
            labels[cluster_id] = "Low-Risk Baseline"
        else:
            labels[cluster_id] = "Intermediate Risk Group"

    return labels


def main():
    print("Loading segment-level cluster dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("\nLoaded shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    required_cols = {"segment_id", "avg_crash_rate", "total_crashes"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in GMM input: {missing}")

    non_feature_cols = {"segment_id", "road_type"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    X = df[feature_cols].copy()

    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns still present in GMM features: {non_numeric}")

    print("\nNumber of GMM features:", len(feature_cols))
    print("Feature columns:")
    print(feature_cols)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print("\nTraining GMM...")
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=10,
    )
    gmm.fit(X_scaled)

    cluster_ids = gmm.predict(X_scaled)
    cluster_probs = gmm.predict_proba(X_scaled).max(axis=1)

    result = df.copy()
    result["gmm_cluster"] = cluster_ids
    result["cluster_probability_max"] = cluster_probs

    summary_aggs = {}
    for col in [
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
    ]:
        if col in result.columns:
            summary_aggs[col] = "mean"

    for col in [
        "oneway",
        "is_bridge",
        "is_tunnel",
        "is_junction",
        "near_intersection",
        "near_traffic_signal",
        "near_crossing",
    ]:
        if col in result.columns:
            summary_aggs[col] = "mean"

    summary = (
        result.groupby("gmm_cluster")
        .agg(
            n_segments=("segment_id", "count"),
            **{
                col: (col, agg_func)
                for col, agg_func in summary_aggs.items()
            },
        )
        .reset_index()
    )

    label_map = build_cluster_label_map(summary)
    result["cluster_label"] = result["gmm_cluster"].map(label_map)
    summary["cluster_label"] = summary["gmm_cluster"].map(label_map)

    if "road_type" in result.columns:
        road_mix = (
            result.groupby(["gmm_cluster", "road_type"])
            .size()
            .reset_index(name="count")
        )
        top_road = (
            road_mix.sort_values(["gmm_cluster", "count"], ascending=[True, False])
            .groupby("gmm_cluster")
            .head(1)[["gmm_cluster", "road_type"]]
            .rename(columns={"road_type": "dominant_road_type"})
        )
        summary = summary.merge(top_road, on="gmm_cluster", how="left")

    sil = silhouette_score(X_scaled, cluster_ids)
    cluster_counts = result["gmm_cluster"].value_counts().sort_index().to_dict()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_ASSIGNMENTS_PATH, index=False)
    summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    joblib.dump(gmm, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics = {
        "input_path": str(INPUT_PATH),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "n_clusters": int(N_CLUSTERS),
        "silhouette_score": float(sil),
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "assignments_path": str(OUTPUT_ASSIGNMENTS_PATH),
        "summary_path": str(OUTPUT_SUMMARY_PATH),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved assignments to:", OUTPUT_ASSIGNMENTS_PATH)
    print("Saved summary to:", OUTPUT_SUMMARY_PATH)
    print("Saved metrics to:", METRICS_PATH)

    print("\nSilhouette score:", round(sil, 4))
    print("\nCluster counts:")
    print(result["gmm_cluster"].value_counts().sort_index())

    print("\nCluster summary:")
    print(summary)


if __name__ == "__main__":
    main()