from pathlib import Path
import json

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


INPUT_PATH = Path("data/processed/full_panel_features_with_infra.csv")
MODEL_DIR = Path("models")
METRICS_PATH = MODEL_DIR / "lightgbm_full_panel_with_infra_metrics.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "lightgbm_full_panel_with_infra_feature_importance.csv"

TRAIN_END_DATE = "2022-12-31"
VALID_END_DATE = "2023-12-31"

TARGET_COL = "crash_occurred"
NON_FEATURE_COLS = {"segment_id", "date", "crash_count"}

THRESHOLD = 0.5

# Set to an integer like 2_000_000 for a faster trial run if needed
SAMPLE_N = None


def temporal_split(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    train_df = df[df["date"] <= TRAIN_END_DATE].copy()
    valid_df = df[(df["date"] > TRAIN_END_DATE) & (df["date"] <= VALID_END_DATE)].copy()
    test_df = df[df["date"] > VALID_END_DATE].copy()

    print("\nTemporal split:")
    print(f"Train rows: {len(train_df):,}")
    print(f"Valid rows: {len(valid_df):,}")
    print(f"Test rows:  {len(test_df):,}")

    return train_df, valid_df, test_df


def prepare_xy(df: pd.DataFrame):
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c != TARGET_COL
    ]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).copy()

    if "segment_id" in X.columns:
        raise ValueError("segment_id STILL in feature set — something is wrong")

    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")

    return X, y, feature_cols


def get_scale_pos_weight(y_train: pd.Series) -> float:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())

    if positives == 0:
        return 1.0

    return negatives / positives


def evaluate(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def top_k_capture(y_true, y_prob, top_fraction: float) -> dict:
    n = len(y_true)
    k = max(1, int(n * top_fraction))

    ranked = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).sort_values(
        "y_prob", ascending=False
    )

    top = ranked.head(k)

    precision_at_k = float(top["y_true"].mean())
    total_positives = int(ranked["y_true"].sum())
    captured_positives = int(top["y_true"].sum())
    recall_at_k = float(captured_positives / total_positives) if total_positives > 0 else 0.0

    return {
        "top_fraction": top_fraction,
        "k_rows": k,
        "precision_at_k": precision_at_k,
        "captured_positives": captured_positives,
        "total_positives": total_positives,
        "recall_at_k": recall_at_k,
    }


def main():
    print("Loading enriched full panel...")
    df = pd.read_csv(INPUT_PATH)

    if SAMPLE_N is not None and SAMPLE_N < len(df):
        print(f"\nSampling {SAMPLE_N:,} rows for faster experimentation...")
        df = df.sample(n=SAMPLE_N, random_state=42).copy()

    print("\nLoaded shape:", df.shape)
    print("\nDataset columns:")
    print(df.columns.tolist())

    print("\nInfrastructure feature presence:")
    infra_features_to_check = [
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
    for col in infra_features_to_check:
        print(f"{col}: {'FOUND' if col in df.columns else 'MISSING'}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = temporal_split(df)

    X_train, y_train, feature_cols = prepare_xy(train_df)
    X_valid, y_valid, _ = prepare_xy(valid_df)
    X_test, y_test, _ = prepare_xy(test_df)

    print("\nFinal feature columns used:")
    print(feature_cols)

    print("\nFeature matrix shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_valid: {X_valid.shape}")
    print(f"X_test:  {X_test.shape}")

    print("\nTarget rates:")
    print(f"Train positive rate: {y_train.mean():.6f}")
    print(f"Valid positive rate: {y_valid.mean():.6f}")
    print(f"Test positive rate:  {y_test.mean():.6f}")

    scale_pos_weight = get_scale_pos_weight(y_train)
    print(f"\nscale_pos_weight: {scale_pos_weight:.4f}")

    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print("\nTraining LightGBM...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
    )

    print("Scoring validation and test...")
    y_valid_prob = model.predict_proba(X_valid)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    valid_metrics = evaluate(y_valid, y_valid_prob, THRESHOLD)
    test_metrics = evaluate(y_test, y_test_prob, THRESHOLD)

    valid_top_1pct = top_k_capture(y_valid.to_numpy(), y_valid_prob, 0.01)
    valid_top_5pct = top_k_capture(y_valid.to_numpy(), y_valid_prob, 0.05)
    test_top_1pct = top_k_capture(y_test.to_numpy(), y_test_prob, 0.01)
    test_top_5pct = top_k_capture(y_test.to_numpy(), y_test_prob, 0.05)

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print("\nTop 30 features:")
    print(feature_importance.head(30))

    important_infra = [
        "segment_curvature",
        "bearing_change_max",
        "intersection_degree_max",
        "near_intersection",
        "near_traffic_signal",
        "lanes",
        "maxspeed",
        "visibility_risk_score",
    ]

    print("\nInfrastructure feature ranks:")
    fi_ranked = feature_importance.reset_index()
    fi_ranked["rank"] = fi_ranked["index"] + 1
    for feat in important_infra:
        if feat in fi_ranked["feature"].values:
            rank = int(fi_ranked.loc[fi_ranked["feature"] == feat, "rank"].iloc[0])
            imp = float(fi_ranked.loc[fi_ranked["feature"] == feat, "importance"].iloc[0])
            print(f"{feat}: rank {rank}, importance {imp}")
        else:
            print(f"{feat}: not found")

    print("\nValidation metrics")
    print(f"ROC-AUC: {valid_metrics['roc_auc']:.4f}")
    print(f"Average Precision: {valid_metrics['average_precision']:.4f}")
    print(f"Balanced Accuracy: {valid_metrics['balanced_accuracy']:.4f}")
    print(f"F1: {valid_metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(valid_metrics["confusion_matrix"]))

    print("\nTest metrics")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Average Precision: {test_metrics['average_precision']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(test_metrics["confusion_matrix"]))

    output = {
        "model": "LightGBM Full Panel With Infrastructure",
        "input_path": str(INPUT_PATH),
        "train_end_date": TRAIN_END_DATE,
        "valid_end_date": VALID_END_DATE,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "n_features": int(X_train.shape[1]),
        "feature_columns": list(feature_cols),
        "scale_pos_weight": float(scale_pos_weight),
        "threshold": float(THRESHOLD),
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "validation_top_1pct": valid_top_1pct,
        "validation_top_5pct": valid_top_5pct,
        "test_top_1pct": test_top_1pct,
        "test_top_5pct": test_top_5pct,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved metrics to {METRICS_PATH}")
    print(f"Saved feature importance to {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()