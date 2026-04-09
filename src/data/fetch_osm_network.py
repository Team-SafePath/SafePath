from pathlib import Path

import geopandas as gpd
import osmnx as ox


PLACE_NAME = "New York City, New York, USA"

EDGES_OUT = Path("data/raw/nyc_street_network.geojson")
NODES_OUT = Path("data/raw/nyc_street_nodes.geojson")


def main():
    print(f"Downloading drivable street network for {PLACE_NAME}...")
    G = ox.graph_from_place(PLACE_NAME, network_type="drive", simplify=True)

    print("Converting graph to GeoDataFrames...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    nodes_gdf = nodes_gdf.reset_index()
    edges_gdf = edges_gdf.reset_index()

    # Make sure CRS is set for web/geo work
    if nodes_gdf.crs is None:
        nodes_gdf = nodes_gdf.set_crs(epsg=4326)
    if edges_gdf.crs is None:
        edges_gdf = edges_gdf.set_crs(epsg=4326)

    # Keep a richer set of edge columns if they exist
    edge_keep = [
        "u",
        "v",
        "key",
        "osmid",
        "name",
        "highway",
        "oneway",
        "reversed",
        "length",
        "lanes",
        "maxspeed",
        "bridge",
        "tunnel",
        "junction",
        "access",
        "service",
        "geometry",
    ]
    edge_keep = [c for c in edge_keep if c in edges_gdf.columns]
    edges_gdf = edges_gdf[edge_keep].copy()

    # Keep a useful set of node columns if they exist
    node_keep = [
        "osmid",
        "highway",
        "street_count",
        "traffic_signals",
        "crossing",
        "railway",
        "junction",
        "geometry",
    ]
    node_keep = [c for c in node_keep if c in nodes_gdf.columns]
    nodes_gdf = nodes_gdf[node_keep].copy()

    EDGES_OUT.parent.mkdir(parents=True, exist_ok=True)
    NODES_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving edges to {EDGES_OUT}...")
    edges_gdf.to_file(EDGES_OUT, driver="GeoJSON")

    print(f"Saving nodes to {NODES_OUT}...")
    nodes_gdf.to_file(NODES_OUT, driver="GeoJSON")

    print("\nDone.")
    print(f"Edges: {len(edges_gdf):,}")
    print(f"Nodes: {len(nodes_gdf):,}")
    print("\nEdge columns:")
    print(edges_gdf.columns.tolist())
    print("\nNode columns:")
    print(nodes_gdf.columns.tolist())


if __name__ == "__main__":
    main()