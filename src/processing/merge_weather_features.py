from pathlib import Path
import pandas as pd


MODEL_PATH = Path("data/processed/modeling_dataset_sampled.csv")
WEATHER_PATH = Path("data/raw/nyc_weather_2018_2024.csv")

OUTPUT_PATH = Path("data/processed/modeling_dataset_with_weather.csv")


def merge_weather_features():

    print("Loading modeling dataset...")
    df = pd.read_csv(MODEL_PATH)

    print(f"Rows: {len(df):,}")

    print("Loading weather dataset...")
    weather = pd.read_csv(WEATHER_PATH)

    df["date"] = pd.to_datetime(df["date"])
    weather["date"] = pd.to_datetime(weather["date"])

    print("Merging weather features...")

    merged = df.merge(
        weather,
        on="date",
        how="left"
    )

    print("Checking for missing weather values...")
    print(merged.isna().sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved dataset with weather features to {OUTPUT_PATH}")

    return merged


if __name__ == "__main__":
    merge_weather_features()