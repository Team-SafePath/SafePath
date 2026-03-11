from pathlib import Path
import requests
import pandas as pd


OUTPUT_PATH = Path("data/raw/nyc_weather_2018_2024.csv")

LAT = 40.7128
LON = -74.0060

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"


def fetch_weather_data():

    print("Fetching weather data from Open-Meteo...")

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": [
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max"
        ],
        "timezone": "America/New_York"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()["daily"]

    df = pd.DataFrame(data)

    df.rename(columns={"time": "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved weather data to {OUTPUT_PATH}")
    print(df.head())

    return df


if __name__ == "__main__":
    fetch_weather_data()