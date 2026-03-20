from pathlib import Path
import json

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


INPUT_PATH = Path("data/processed/modeling_dataset_final.csv")
MODEL_DIR = Path("models")
METRICS_PATH = MODEL_DIR / "lightgbm_metrics.json"

TARGET_COL = "crash_occurred"


def normalize_road_type(x):
    if pd.isna(x):
        return "unknown"

    x = str(x).strip()

    if x.startswith("[") and x.endswith("]"):
        x = x[1:-1].replace("'", "").replace('"', "").strip()

    if "," in x:
        x = x.split(",")[0].strip()
    else:
        x = x.split()[0].strip() if x else "unknown"

    return x if x else "unknown"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "road_type" in df.columns:
        df["road_type"] = df["road_type"].apply(normalize_road_type)
        road_dummies = pd.get_dummies(df["road_type"], prefix="road", dtype=int)
        df = pd.concat([df.drop(columns=["road_type"]), road_dummies], axis=1)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    return df


def temporal_split(df: pd.DataFrame):
    train_df = df[df["date"] < "2023-01-01"].copy()
    test_df = df[df["date"] >= "2023-01-01"].copy()

    print("\nTemporal split:")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")

    return train_df, test_df


def evaluate_model(model, x_train, x_test, y_train, y_test, model_name: str) -> dict:
    print(f"\nTraining {model_name}...")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    print(f"\n{model_name}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def get_feature_importance(model, feature_names):
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance_df


def run_lightgbm_model(use_lag_features: bool = True):
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    df = prepare_features(df)
    df = df[df["date"].notna()].copy()

    train_df, test_df = temporal_split(df)

    drop_cols = ["date", "segment_id", "crash_count"]
    if not use_lag_features:
        drop_cols.extend(["crashes_last_7_days", "crashes_last_30_days"])

    existing_drop_cols = [col for col in drop_cols if col in train_df.columns]

    x_train = train_df.drop(columns=[TARGET_COL] + existing_drop_cols)
    y_train = train_df[TARGET_COL].astype(int)

    x_test = test_df.drop(columns=[TARGET_COL] + existing_drop_cols)
    y_test = test_df[TARGET_COL].astype(int)

    print(f"Train feature shape: {x_train.shape}")
    print(f"Test feature shape: {x_test.shape}")

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42,
        n_jobs=-1,
    )

    model_name = "LightGBM (with lag)" if use_lag_features else "LightGBM (no lag)"
    metrics = evaluate_model(model, x_train, x_test, y_train, y_test, model_name)

    importance_df = get_feature_importance(model, x_train.columns)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "with_lag" if use_lag_features else "no_lag"
    importance_path = MODEL_DIR / f"lightgbm_feature_importance_{suffix}.csv"

    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importance to {importance_path}")

    return metrics, importance_df


def main():
    with_lag_metrics, _ = run_lightgbm_model(use_lag_features=True)
    no_lag_metrics, _ = run_lightgbm_model(use_lag_features=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {
        "lightgbm_with_lag": with_lag_metrics,
        "lightgbm_no_lag": no_lag_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()