from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/processed/modeling_dataset_final.csv")
OUTPUT_PATH = Path("data/processed/modeling_dataset_ml.csv")


def finalize_model_features():

    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Encoding road types...")
    # Clean road_type first
    def normalize_road_type(x):
        if isinstance(x, str):
            if "[" in x:
                x = x.replace("[", "").replace("]", "").replace("'", "")
                x = x.split()[0]
        return x

    df["road_type"] = df["road_type"].apply(normalize_road_type)

    print("Encoding road types...")
    road_dummies = pd.get_dummies(df["road_type"], prefix="road", dtype=int)
    df = pd.concat([df, road_dummies], axis=1)

    print("Creating rain indicator...")
    df["rain_indicator"] = (df["precipitation_sum"] > 0).astype(int)

    print("Dropping non-model columns...")

    drop_cols = ["date", "road_type", "segment_id", "crash_count"]
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved ML dataset to {OUTPUT_PATH}")
    print(df.head())

    return df


if __name__ == "__main__":
    finalize_model_features()