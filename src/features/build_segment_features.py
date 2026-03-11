from pathlib import Path
import geopandas as gpd


STREETS_PATH = Path("data/raw/nyc_street_network.geojson")
OUTPUT_PATH = Path("data/processed/segment_features.csv")


def build_segment_features():

    print("Loading street network...")
    streets = gpd.read_file(STREETS_PATH)

    streets = streets.reset_index(drop=True)

    # create segment_id consistent with earlier scripts
    streets["segment_id"] = streets.index.astype(str)

    print("Building infrastructure features...")

    streets["segment_length"] = streets["length"]

    streets["road_type"] = streets["highway"].astype(str)

    features = streets[[
        "segment_id",
        "segment_length",
        "road_type"
    ]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved segment features to {OUTPUT_PATH}")

    return features


if __name__ == "__main__":
    build_segment_features()