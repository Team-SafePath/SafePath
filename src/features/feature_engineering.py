from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/processed/modeling_dataset_full.csv")
OUTPUT_PATH = Path("data/processed/modeling_dataset_features.csv")


def feature_engineering():

    print("Loading modeling dataset...")
    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    print("Creating temporal features...")

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    print("Creating weather indicators...")

    df["rain_indicator"] = (df["precipitation_sum"] > 0).astype(int)

    print("Encoding road types...")

    road_dummies = pd.get_dummies(df["road_type"], prefix="road")

    df = pd.concat([df, road_dummies], axis=1)

    print("Dropping unused columns...")

    df = df.drop(columns=["date", "road_type"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved engineered dataset to {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    feature_engineering()