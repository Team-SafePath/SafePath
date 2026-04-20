"use client";

import { useEffect, useMemo, useState } from "react";

type GeoJsonFeature = {
  type: string;
  properties: {
    gmm_cluster?: number;
    cluster_label?: string;
    avg_predicted_risk?: number;
    risk_percentile?: number;
    maxspeed?: number;
    visibility_risk_score?: number;
    intersection_degree_max?: number;
    near_traffic_signal?: number;
  };
  geometry: GeoJSON.Geometry;
};

type GeoJsonCollection = {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
};

type FeatureImportanceRow = {
  feature: string;
  importance: number;
};

const MAP_DATA_URL = process.env.NEXT_PUBLIC_SEGMENT_MAP_URL;
const FEATURE_IMPORTANCE_URL =
  "/data/lightgbm_full_panel_with_infra_feature_importance.csv";

function mean(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function parseCsv(text: string): FeatureImportanceRow[] {
  const lines = text.trim().split("\n");
  if (lines.length <= 1) return [];

  const headers = lines[0].split(",");

  return lines.slice(1).map((line) => {
    const parts = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = parts[i];
    });

    return {
      feature: row.feature,
      importance: Number(row.importance),
    };
  });
}

function speedBand(speed: number) {
  if (speed <= 20) return "20 mph";
  if (speed <= 25) return "25 mph";
  if (speed <= 30) return "30 mph";
  if (speed <= 40) return "40 mph";
  return "50+ mph";
}

function prettyFeatureName(feature: string) {
  switch (feature) {
    case "temperature_2m_mean":
      return "Average temperature";
    case "segment_length":
      return "Segment length";
    case "windspeed_10m_max":
      return "Maximum wind speed";
    case "precipitation_sum":
      return "Precipitation";
    case "segment_curvature":
      return "Segment curvature";
    case "bearing_change_max":
      return "Bearing change";
    case "visibility_risk_score":
      return "Visibility risk";
    case "crashes_last_30_days":
      return "30-day crash history";
    case "day_of_week":
      return "Day of week";
    case "lanes":
      return "Lane count";
    case "maxspeed":
      return "Posted speed";
    case "intersection_degree_max":
      return "Intersection complexity";
    case "near_traffic_signal":
      return "Traffic signal proximity";
    case "crashes_last_7_days":
      return "7-day crash history";
    case "near_intersection":
      return "Intersection proximity";
    default:
      return feature.replaceAll("_", " ");
  }
}

export default function KeyTakeawaysPanel() {
  const [mapData, setMapData] = useState<GeoJsonCollection | null>(null);
  const [featureRows, setFeatureRows] = useState<FeatureImportanceRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  const configError = !MAP_DATA_URL
    ? "NEXT_PUBLIC_SEGMENT_MAP_URL is not defined. Set it in Vercel and .env.local."
    : null;

  useEffect(() => {
    if (!MAP_DATA_URL) return;

    let cancelled = false;

    Promise.all([
      fetch(MAP_DATA_URL).then((res) => {
        if (!res.ok) throw new Error("Failed to load map data");
        return res.json();
      }),
      fetch(FEATURE_IMPORTANCE_URL).then((res) => {
        if (!res.ok) throw new Error("Failed to load feature importance");
        return res.text();
      }),
    ])
      .then(([geojson, csvText]) => {
        if (cancelled) return;
        setMapData(geojson as GeoJsonCollection);
        setFeatureRows(parseCsv(csvText));
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const takeaways = useMemo(() => {
    if (!mapData) return null;

    const features = mapData.features;

    const groupedByCluster = new Map<string, GeoJsonFeature[]>();
    for (const feature of features) {
      const label = feature.properties.cluster_label ?? "Unassigned";
      if (!groupedByCluster.has(label)) {
        groupedByCluster.set(label, []);
      }
      groupedByCluster.get(label)!.push(feature);
    }

    let topCluster = "N/A";
    let topClusterRisk = -1;

    for (const [label, rows] of groupedByCluster.entries()) {
      const avgRisk = mean(
        rows.map((r) => Number(r.properties.risk_percentile ?? 0))
      );
      if (avgRisk > topClusterRisk) {
        topClusterRisk = avgRisk;
        topCluster = label;
      }
    }

    const groupedBySpeed = new Map<string, number[]>();
    for (const feature of features) {
      const speed = Number(feature.properties.maxspeed ?? NaN);
      const risk = Number(feature.properties.risk_percentile ?? NaN);
      if (Number.isNaN(speed) || Number.isNaN(risk)) continue;

      const band = speedBand(speed);
      if (!groupedBySpeed.has(band)) {
        groupedBySpeed.set(band, []);
      }
      groupedBySpeed.get(band)!.push(risk);
    }

    let topSpeedBand = "N/A";
    let topSpeedBandRisk = -1;

    for (const [band, values] of groupedBySpeed.entries()) {
      const avgRisk = mean(values);
      if (avgRisk > topSpeedBandRisk) {
        topSpeedBandRisk = avgRisk;
        topSpeedBand = band;
      }
    }

    const topFeatures = [...featureRows]
      .filter(
        (row) =>
          !["month", "sin_month", "cos_month", "sin_day_of_week", "cos_day_of_week"].includes(
            row.feature
          )
      )
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 3)
      .map((row) => prettyFeatureName(row.feature));

    return {
      topCluster,
      topClusterRisk,
      topSpeedBand,
      topSpeedBandRisk,
      topFeatures,
    };
  }, [mapData, featureRows]);

  const displayError = configError || error;

  if (displayError) {
    return <div className="text-sm text-red-600">{displayError}</div>;
  }

  if (!takeaways) {
    return <div className="text-sm text-slate-500">Loading key takeaways...</div>;
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6">
      <div className="mb-4">
        <h2 className="text-xl font-semibold">Key Takeaways</h2>
        <p className="mt-1 text-sm text-slate-600">
          A quick summary of the main patterns surfaced by the model and spatial analysis.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">
            Where risk concentrates
          </div>
          <div className="mt-3 text-base font-semibold text-slate-900">
            {takeaways.topCluster}
          </div>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            This cluster shows the highest average modeled risk percentile across the
            network, suggesting a recurring roadway archetype rather than isolated hotspots.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">
            Speed environment
          </div>
          <div className="mt-3 text-base font-semibold text-slate-900">
            {takeaways.topSpeedBand}
          </div>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            This posted speed band has the highest average modeled risk percentile, which
            suggests speed environment is part of the broader crash-risk pattern.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
          <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">
            Strongest model signals
          </div>
          <div className="mt-3 text-base font-semibold text-slate-900">
            {takeaways.topFeatures.join(", ")}
          </div>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            These features were among the strongest contributors in the final LightGBM
            model, reinforcing that roadway structure and local conditions matter alongside
            recent crash history.
          </p>
        </div>
      </div>
    </div>
  );
}