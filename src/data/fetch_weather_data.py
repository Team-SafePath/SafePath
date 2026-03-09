from pathlib import Path
import pandas as pd
import requests

# Output file
OUTPUT_PATH = Path("data/raw/nyc_weather_2018_2024.csv")


# NYC coordinate (Central Park)
LAT = 40.7812
LON = -73.9665

# Potential Future locations
BOROUGH_COORDS = {
    "manhattan": (40.7831, -73.9712),
    "brooklyn": (40.6782, -73.9442),
    "queens": (40.7282, -73.7949),
    "bronx": (40.8448, -73.8648),
    "staten_island": (40.5795, -74.1502),
}

# Date range to match crash dataset
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# Weather variables to request
HOURLY_VARS = ",".join([
    "temperature_2m",
    "relativehumidity_2m",
    "cloudcover",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "windspeed_10m",
    "winddirection_10m",
    "windgusts_10m",
    "weathercode"
])

def fetch_weather_data():
    """
    Fetch hourly historical weather for NYC using the Open-Meteo Archive API.
    Saves results to data/raw/nyc_weather_2018_2024.csv.
    """

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching NYC hourly weather from {START_DATE} to {END_DATE}...")

    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": HOURLY_VARS,
        "timezone": "America/New_York"
    }

    r = requests.get(base_url, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    hourly = payload.get("hourly", {})
    if not hourly:
        print("No hourly weather data returned.")
        return None

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    fetch_weather_data()
    