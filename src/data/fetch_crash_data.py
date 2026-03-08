from pathlib import Path
from io import StringIO

import pandas as pd
import requests


BASE_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.csv"
OUTPUT_PATH = Path("data/raw/nyc_crashes_2018_2024.csv")

CHUNK_SIZE = 50000
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"


def fetch_crash_data(limit: int = 2000000):

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    offset = 0

    while offset < limit:

        params = {
            "$limit": CHUNK_SIZE,
            "$offset": offset,
            "$order": "crash_date ASC",
            "$where": f"crash_date between '{START_DATE}' and '{END_DATE}'"
        }

        print(f"Fetching rows {offset} to {offset + CHUNK_SIZE - 1}...")

        r = requests.get(BASE_URL, params=params, timeout=60)
        r.raise_for_status()

        chunk = pd.read_csv(StringIO(r.text))

        if chunk.empty:
            print("No more rows returned by API.")
            break

        all_chunks.append(chunk)

        if len(chunk) < CHUNK_SIZE:
            print("Reached end of filtered dataset.")
            break

        offset += CHUNK_SIZE

    df = pd.concat(all_chunks, ignore_index=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    fetch_crash_data()