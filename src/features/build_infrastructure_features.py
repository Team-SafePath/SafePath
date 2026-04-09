from pathlib import Path
import ast
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point


EDGES_PATH = Path("data/raw/nyc_street_network.geojson")
NODES_PATH = Path("data/raw/nyc_street_nodes.geojson")
OUT_PATH = Path("data/processed/segment_infrastructure_features.csv")


def parse_list_like(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (tuple, set)):
        return list(val)
    if isinstance(val, str):
        text = val.strip()
        if not text or text.lower() == "nan":
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return [text]
    if pd.isna(val):
        return []
    return [val]


def first_numeric(val):
    vals = parse_list_like(val)
    for item in vals:
        if item is None:
            continue
        s = str(item).strip().lower().replace("mph", "").replace("km/h", "")
        try:
            return float(s)
        except Exception:
            continue
    return np.nan


def normalize_bool(val):
    vals = [str(v).strip().lower() for v in parse_list_like(val)]
    if not vals:
        return 0
    true_set = {"yes", "true", "1", "reversible", "y"}
    return int(any(v in true_set for v in vals))


def normalize_road_type(val):
    vals = [str(v).strip().lower() for v in parse_list_like(val)]
    vals = [v for v in vals if v]
    if not vals:
        return "unknown"
    vals = sorted(set(vals))
    return " ".join(vals[:3])


def coords_from_geom(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return list(geom.coords)
    if isinstance(geom, MultiLineString):
        coords = []
        for part in geom.geoms:
            coords.extend(list(part.coords))
        return coords
    return []


def segment_curvature(geom):
    coords = coords_from_geom(geom)
    if len(coords) < 3:
        return 0.0

    line = LineString(coords)
    chord = Point(coords[0]).distance(Point(coords[-1]))
    if chord == 0:
        return 0.0

    return max((line.length / chord) - 1.0, 0.0)


def bearing(p1, p2):
    lon1, lat1 = p1
    lon2, lat2 = p2
    angle = math.degrees(math.atan2(lat2 - lat1, lon2 - lon1))
    return (angle + 360) % 360


def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def max_bearing_change(geom):
    coords = coords_from_geom(geom)
    if len(coords) < 3:
        return 0.0

    bearings = []
    for i in range(len(coords) - 1):
        if coords[i] != coords[i + 1]:
            bearings.append(bearing(coords[i], coords[i + 1]))

    if len(bearings) < 2:
        return 0.0

    changes = [
        angle_diff(bearings[i], bearings[i + 1])
        for i in range(len(bearings) - 1)
    ]
    return float(max(changes)) if changes else 0.0


def nearest_node_flags(edges_gdf, nodes_gdf):
    if "osmid" not in nodes_gdf.columns:
        raise ValueError("nodes GeoJSON must include 'osmid'")

    nodes_lookup = nodes_gdf.drop_duplicates("osmid").set_index("osmid")

    def node_feature(osmid, col, default=np.nan):
        if osmid in nodes_lookup.index and col in nodes_lookup.columns:
            return nodes_lookup.at[osmid, col]
        return default

    out = pd.DataFrame(index=edges_gdf.index)

    out["u_degree"] = edges_gdf["u"].map(
        lambda x: node_feature(x, "street_count", np.nan)
    )
    out["v_degree"] = edges_gdf["v"].map(
        lambda x: node_feature(x, "street_count", np.nan)
    )

    out["intersection_degree_max"] = out[["u_degree", "v_degree"]].max(axis=1)
    out["near_intersection"] = (
        out["intersection_degree_max"].fillna(0).ge(4).astype(int)
    )

    def node_has_signal(osmid):
        val = node_feature(osmid, "highway", "")
        vals = [str(v).strip().lower() for v in parse_list_like(val)]
        return int("traffic_signals" in vals)

    def node_has_crossing(osmid):
        crossing = str(node_feature(osmid, "crossing", "")).strip().lower()
        highway = node_feature(osmid, "highway", "")
        highway_vals = [str(v).strip().lower() for v in parse_list_like(highway)]
        return int(crossing not in {"", "nan", "none"} or "crossing" in highway_vals)

    out["near_traffic_signal"] = edges_gdf.apply(
        lambda r: max(node_has_signal(r["u"]), node_has_signal(r["v"])),
        axis=1,
    )
    out["near_crossing"] = edges_gdf.apply(
        lambda r: max(node_has_crossing(r["u"]), node_has_crossing(r["v"])),
        axis=1,
    )

    return out


def main():
    print("Loading edges...")
    edges = gpd.read_file(EDGES_PATH).reset_index(drop=True)
    edges["segment_id"] = edges.index.astype(int)

    print("Loading nodes...")
    nodes = gpd.read_file(NODES_PATH).reset_index(drop=True)

    if edges.crs is not None and nodes.crs is not None and edges.crs != nodes.crs:
        nodes = nodes.to_crs(edges.crs)

    print("Computing segment-level infrastructure features...")
    df = pd.DataFrame()
    df["segment_id"] = edges["segment_id"]

    df["road_type"] = (
        edges["highway"].apply(normalize_road_type)
        if "highway" in edges.columns
        else "unknown"
    )
    df["segment_length"] = (
        edges["length"].fillna(0.0) if "length" in edges.columns else 0.0
    )
    df["oneway"] = (
        edges["oneway"].apply(normalize_bool) if "oneway" in edges.columns else 0
    )
    df["lanes"] = (
        edges["lanes"].apply(first_numeric) if "lanes" in edges.columns else np.nan
    )
    df["maxspeed"] = (
        edges["maxspeed"].apply(first_numeric)
        if "maxspeed" in edges.columns
        else np.nan
    )
    df["is_bridge"] = (
        edges["bridge"].apply(normalize_bool) if "bridge" in edges.columns else 0
    )
    df["is_tunnel"] = (
        edges["tunnel"].apply(normalize_bool) if "tunnel" in edges.columns else 0
    )
    df["is_junction"] = (
        edges["junction"].notna().astype(int) if "junction" in edges.columns else 0
    )

    df["segment_curvature"] = edges.geometry.apply(segment_curvature)
    df["bearing_change_max"] = edges.geometry.apply(max_bearing_change)

    df["segment_curvature"] = df["segment_curvature"].clip(lower=0, upper=5)

    if {"u", "v"}.issubset(edges.columns):
        node_flags = nearest_node_flags(edges, nodes)
        df = pd.concat([df, node_flags.reset_index(drop=True)], axis=1)
    else:
        df["u_degree"] = np.nan
        df["v_degree"] = np.nan
        df["intersection_degree_max"] = np.nan
        df["near_intersection"] = 0
        df["near_traffic_signal"] = 0
        df["near_crossing"] = 0

    curvature_max = max(df["segment_curvature"].max(), 1e-6)
    bearing_max = max(df["bearing_change_max"].max(), 1e-6)
    intersection_max = max(df["intersection_degree_max"].max(), 1e-6)

    df["curvature_norm"] = df["segment_curvature"] / curvature_max
    df["bearing_norm"] = df["bearing_change_max"] / bearing_max
    df["intersection_norm"] = df["intersection_degree_max"] / intersection_max

    df["visibility_risk_score"] = (
        0.4 * df["curvature_norm"]
        + 0.4 * df["bearing_norm"]
        + 0.2 * df["intersection_norm"]
    ).clip(lower=0, upper=1)

    for col in [
        "lanes",
        "maxspeed",
        "u_degree",
        "v_degree",
        "intersection_degree_max",
        "segment_length",
        "segment_curvature",
        "bearing_change_max",
        "curvature_norm",
        "bearing_norm",
        "intersection_norm",
        "visibility_risk_score",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved infrastructure features to {OUT_PATH}")
    print(f"Rows: {len(df):,}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nHead:")
    print(df.head(10))

    print("\nNull counts:")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nValue counts:")
    print(df["near_intersection"].value_counts(dropna=False))
    print(df["near_traffic_signal"].value_counts(dropna=False))
    print(df["near_crossing"].value_counts(dropna=False))

    print("\nVisibility risk summary:")
    print(df["visibility_risk_score"].describe())


if __name__ == "__main__":
    main()