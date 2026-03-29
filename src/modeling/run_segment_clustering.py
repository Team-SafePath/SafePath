import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

INPUT_PATH = "data/processed/segment_profiles.csv"
OUTPUT_PATH = "data/processed/segment_clusters.csv"


def run_clustering(n_clusters=5):
    print("Loading segment profiles...")
    df = pd.read_csv(INPUT_PATH)

    # Drop non-feature columns
    drop_cols = [
        "segment_id",
        "avg_temp",
        "avg_precip",
        "avg_wind"
    ]

    X = df.drop(columns=drop_cols)

    print("Feature shape:", X.shape)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # KMEANS
    # -------------------------
    print("Running KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

    # -------------------------
    # GMM (MAIN MODEL)
    # -------------------------
    print("Running Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    df["gmm_cluster"] = gmm.fit_predict(X_scaled)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved clusters to {OUTPUT_PATH}")

    # Print cluster sizes
    print("\nKMeans cluster counts:")
    print(df["kmeans_cluster"].value_counts())

    print("\nGMM cluster counts:")
    print(df["gmm_cluster"].value_counts())


if __name__ == "__main__":
    run_clustering(n_clusters=5)