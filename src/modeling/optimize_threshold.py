import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

PRED_PATH = "data/processed/full_panel_predictions.csv"
OUTPUT_PATH = "models/full_panel_threshold_results.csv"


def top_k_capture(y_true, y_prob, top_fraction):
    n = len(y_true)
    k = max(1, int(n * top_fraction))

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False)

    top = df.head(k)

    precision_at_k = top["y_true"].mean()
    total_positives = df["y_true"].sum()
    captured = top["y_true"].sum()

    recall_at_k = captured / total_positives if total_positives > 0 else 0

    return precision_at_k, recall_at_k


def optimize_threshold():
    print("Loading predictions...")
    df = pd.read_csv(PRED_PATH)

    if "crash_occurred" not in df.columns or "predicted_proba" not in df.columns:
        raise ValueError("Missing required columns in predictions file")

    y_true = df["crash_occurred"].values
    y_prob = df["predicted_proba"].values

    # Clean NaNs just in case
    mask = ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    thresholds = [
        0.01, 0.02, 0.03, 0.04, 0.05,
        0.075, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50
    ]

    results = []

    print("Evaluating thresholds...\n")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    results_df = pd.DataFrame(results)

    # Sort for readability
    results_df = results_df.sort_values("f1", ascending=False)

    # Save results
    results_df.to_csv(OUTPUT_PATH, index=False)

    best = results_df.iloc[0]

    print("Best threshold (by F1):")
    print(best)

    # 🔥 Top-K metrics (VERY important for your project)
    print("\nTop-K metrics (ranking performance):")

    for frac in [0.01, 0.05]:
        p, r = top_k_capture(y_true, y_prob, frac)
        print(f"Top {int(frac*100)}% → Precision: {p:.4f}, Recall: {r:.4f}")

    print(f"\nSaved threshold results to: {OUTPUT_PATH}")

    return results_df


if __name__ == "__main__":
    optimize_threshold()