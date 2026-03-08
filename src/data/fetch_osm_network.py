from pathlib import Path

import osmnx as ox


OUTPUT_PATH = Path("data/raw/nyc_street_network.geojson")


def fetch_osm_network():
    print("Downloading NYC street network from OpenStreetMap...")

    G = ox.graph_from_place(
        "New York City, New York, USA",
        network_type="drive",
        simplify=True,
    )

    print("Converting graph to GeoDataFrame...")
    nodes, edges = ox.graph_to_gdfs(G)

    # In OSMnx, edge IDs are often stored in the index: (u, v, key)
    edges = edges.reset_index()

    keep_cols = ["u", "v", "key", "length", "highway", "name", "geometry"]
    existing_cols = [col for col in keep_cols if col in edges.columns]
    edges = edges[existing_cols].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    edges.to_file(OUTPUT_PATH, driver="GeoJSON")

    print(f"Saved street network to {OUTPUT_PATH}")
    print(f"Total street segments: {len(edges):,}")

    return edges


if __name__ == "__main__":
    fetch_osm_network()