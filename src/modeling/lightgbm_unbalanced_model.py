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


INPUT_PATH = Path("data/processed/lstm/lstm_panel.csv")
MODEL_DIR = Path("models")
METRICS_PATH = MODEL_DIR / "lightgbm_unbalanced_metrics.json"

TARGET_COL = "crash_occurred"
TRAIN_END_DATE = "2022-12-31"
VALID_END_DATE = "2023-12-31"


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


def prepare_xy(df: pd.DataFrame, use_lag_features: bool):
    drop_cols = ["date", "segment_id"]

    # For this panel, crash_count is allowed as a temporal input.
    # It represents previous observed daily counts within the panel,
    # not the target itself for the same row sequence context.
    # For one-row tabular modeling, though, crash_count would leak,
    # so we drop it here.
    drop_cols.append("crash_count")

    if not use_lag_features:
        drop_cols.extend(["crashes_last_7_days", "crashes_last_30_days"])

    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=[TARGET_COL] + existing_drop_cols)
    y = df[TARGET_COL].astype(int)

    return X, y


def evaluate_model(model, X, y, model_name: str) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y, y_prob),
        "average_precision": average_precision_score(y, y_prob),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(
            y,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    print(f"\n{model_name}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    return metrics


def get_feature_importance(model, feature_names):
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_one_model(train_df, valid_df, test_df, use_lag_features: bool):
    X_train, y_train = prepare_xy(train_df, use_lag_features=use_lag_features)
    X_valid, y_valid = prepare_xy(valid_df, use_lag_features=use_lag_features)
    X_test, y_test = prepare_xy(test_df, use_lag_features=use_lag_features)

    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()
    scale_pos_weight = negatives / positives

    print(f"\nUsing lag features: {use_lag_features}")
    print(f"Train positive rate: {y_train.mean():.6f}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"Train feature shape: {X_train.shape}")
    print(f"Valid feature shape: {X_valid.shape}")
    print(f"Test feature shape:  {X_test.shape}")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[],
    )

    model_name = "LightGBM (with lag)" if use_lag_features else "LightGBM (no lag)"

    valid_metrics = evaluate_model(model, X_valid, y_valid, f"{model_name} - validation")
    test_metrics = evaluate_model(model, X_test, y_test, f"{model_name} - test")

    importance_df = get_feature_importance(model, X_train.columns)

    suffix = "with_lag" if use_lag_features else "no_lag"
    importance_path = MODEL_DIR / f"lightgbm_unbalanced_feature_importance_{suffix}.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importance to {importance_path}")

    return {
        "scale_pos_weight": float(scale_pos_weight),
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }


def main():
    print("Loading full unbalanced panel...")
    df = pd.read_csv(INPUT_PATH)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = temporal_split(df)

    with_lag_results = train_one_model(
        train_df,
        valid_df,
        test_df,
        use_lag_features=True,
    )

    no_lag_results = train_one_model(
        train_df,
        valid_df,
        test_df,
        use_lag_features=False,
    )

    all_metrics = {
        "lightgbm_unbalanced_with_lag": with_lag_results,
        "lightgbm_unbalanced_no_lag": no_lag_results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()