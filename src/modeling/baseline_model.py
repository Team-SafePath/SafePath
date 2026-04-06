from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
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
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
RF_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
FEATURE_IMPORTANCE_PLOT_PATH = MODEL_DIR / "random_forest_feature_importance.png"

TARGET_COL = "crash_occurred"
DROP_COLS = ["date", "crash_count", "segment_id"]


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

    print(f"\n{model_name} Results")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


def plot_random_forest_feature_importance(model, feature_names):
    rf = model.named_steps["classifier"]
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 7))
    top_features.sort_values().plot(kind="barh")
    plt.title("Random Forest Feature Importance (Top 20)")
    plt.xlabel("Importance")
    plt.tight_layout()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved feature importance plot to {FEATURE_IMPORTANCE_PLOT_PATH}")


def train_baseline_models():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    df = prepare_features(df)
    df = df[df["date"].notna()].copy()

    print(f"Dataset shape after preparation: {df.shape}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    train_df, test_df = temporal_split(df)

    x_train = train_df.drop(
        columns=[TARGET_COL, "date"] + [col for col in DROP_COLS if col in train_df.columns]
    )
    y_train = train_df[TARGET_COL].astype(int)

    x_test = test_df.drop(
        columns=[TARGET_COL, "date"] + [col for col in DROP_COLS if col in test_df.columns]
    )
    y_test = test_df[TARGET_COL].astype(int)

    print(f"Train feature shape: {x_train.shape}")
    print(f"Test feature shape: {x_test.shape}")

    logistic_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
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

    logistic_metrics = evaluate_model(
        logistic_pipeline,
        x_train,
        x_test,
        y_train,
        y_test,
        "Logistic Regression",
    )

    rf_metrics = evaluate_model(
        rf_pipeline,
        x_train,
        x_test,
        y_train,
        y_test,
        "Random Forest",
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSaving trained models...")
    joblib.dump(logistic_pipeline, LOGISTIC_MODEL_PATH)
    joblib.dump(rf_pipeline, RF_MODEL_PATH)

    print(f"Saved logistic regression model to {LOGISTIC_MODEL_PATH}")
    print(f"Saved random forest model to {RF_MODEL_PATH}")

    print("\nSaving metrics...")
    all_metrics = {
        "logistic_regression": logistic_metrics,
        "random_forest": rf_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Saved metrics to {METRICS_PATH}")

    plot_random_forest_feature_importance(rf_pipeline, x_train.columns)

    return all_metrics


if __name__ == "__main__":
    train_baseline_models()