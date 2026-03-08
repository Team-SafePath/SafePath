from pathlib import Path

import pandas as pd
import geopandas as gpd


RAW_PATH = Path("data/raw/nyc_crashes_2018_2024.csv")
OUTPUT_PATH = Path("data/interim/nyc_crashes_clean.geojson")


def clean_crash_data():

    print("Loading crash data...")
    df = pd.read_csv(RAW_PATH)

    print(f"Initial rows: {len(df):,}")

    # Convert crash date
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")

    # Drop rows without coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    print(f"Rows after removing missing coordinates: {len(df):,}")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(OUTPUT_PATH, driver="GeoJSON")

    print(f"Saved cleaned crashes to {OUTPUT_PATH}")

    return gdf


if __name__ == "__main__":
    clean_crash_data()