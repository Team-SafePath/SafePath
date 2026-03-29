import pandas as pd
import lightgbm as lgb
import os

INPUT_PATH = "data/processed/full_panel_features.csv"
MODEL_PATH = "models/lightgbm_full_panel.txt"
OUTPUT_PATH = "data/processed/full_panel_predictions.csv"

TARGET_COL = "crash_occurred"
NON_FEATURE_COLS = ["segment_id", "date", "crash_count", TARGET_COL]


def generate_predictions():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)

    print("Loading model...")
    model = lgb.Booster(model_file=MODEL_PATH)

    # Drop non-feature columns safely
    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Ensure all features are numeric
    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}")

    print(f"Feature matrix shape: {X.shape}")

    print("Generating predictions...")
    df["predicted_proba"] = model.predict(X)

    # Keep useful columns for downstream steps (IMPORTANT)
    output_cols = ["segment_id", "date", TARGET_COL, "predicted_proba"]
    output_df = df[[c for c in output_cols if c in df.columns]]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved predictions to {OUTPUT_PATH}")
    print("\nPreview:")
    print(output_df.head())


if __name__ == "__main__":
    generate_predictions()