from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed/modeling_dataset_final.csv")
MODEL_DIR = Path("models")
METRICS_PATH = MODEL_DIR / "baseline_no_lag_metrics.json"

TARGET_COL = "crash_occurred"
DROP_COLS = [
    "date",
    "crash_count",
    "segment_id",
    "crashes_last_7_days",
    "crashes_last_30_days",
]


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

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    for col in numeric_cols:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    return df


def temporal_split(df: pd.DataFrame):
    train = df[df["date"] < "2023-01-01"].copy()
    test = df[df["date"] >= "2023-01-01"].copy()

    print("\nTemporal split:")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")

    return train, test


def evaluate(model, x_train, x_test, y_train, y_test, name):
    print(f"\nTraining {name}...")

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

    print(f"\n{name}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Avg Precision: {metrics['average_precision']:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


def run():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    df = prepare_features(df)
    df = df[df["date"].notna()].copy()

    train_df, test_df = temporal_split(df)

    x_train = train_df.drop(
        columns=[TARGET_COL, "date"] + [c for c in DROP_COLS if c in train_df.columns]
    )
    y_train = train_df[TARGET_COL].astype(int)

    x_test = test_df.drop(
        columns=[TARGET_COL, "date"] + [c for c in DROP_COLS if c in test_df.columns]
    )
    y_test = test_df[TARGET_COL].astype(int)

    print(f"Train feature shape: {x_train.shape}")
    print(f"Test feature shape: {x_test.shape}")

    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_metrics = evaluate(
        logistic,
        x_train,
        x_test,
        y_train,
        y_test,
        "Logistic Regression (no lag)",
    )

    rf_metrics = evaluate(
        rf,
        x_train,
        x_test,
        y_train,
        y_test,
        "Random Forest (no lag)",
    )

    MODEL_DIR.mkdir(exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "logistic_no_lag": logistic_metrics,
                "rf_no_lag": rf_metrics,
            },
            f,
            indent=2,
        )

    print(f"\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    run()