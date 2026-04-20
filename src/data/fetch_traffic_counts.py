import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path

OUTPUT_PATH = Path("data/raw/nyc_traffic_counts.geojson")

API_URL = "https://data.cityofnewyork.us/resource/p424-amsu.json"

def fetch_traffic_counts(limit=5000):
    print("Fetching NYC DOT traffic volume counts...")

    params = {
        "$limit": limit
    }

    r = requests.get(API_URL, params=params)
    r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data)

    # Drop rows without coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude.astype(float), df.latitude.astype(float)),
        crs="EPSG:4326"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT_PATH, driver="GeoJSON")

    print(f"Saved traffic counts to {OUTPUT_PATH}")
    return gdf

if __name__ == "__main__":
    fetch_traffic_counts()
