"use client";

import { useEffect, useMemo, useState } from "react";

type GeoJsonFeature = {
  type: string;
  properties: {
    segment_id?: number;
    gmm_cluster?: number;
    cluster_label?: string;
    road_type?: string;
    total_crashes?: number;
    avg_predicted_risk?: number;
    risk_percentile?: number;
    lanes?: number;
    maxspeed?: number;
    visibility_risk_score?: number;
    segment_curvature?: number;
    bearing_change_max?: number;
    intersection_degree_max?: number;
    near_traffic_signal?: number;
  };
  geometry: GeoJSON.Geometry;
};

type GeoJsonCollection = {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
};

type ClusterSummary = {
  cluster_label: string;
  values: Record<string, number>;
};

type SpeedBandSummary = {
  label: string;
  values: number[];
  count: number;
  mean: number;
  median: number;
  min: number;
  max: number;
};

const MAP_DATA_URL = process.env.NEXT_PUBLIC_SEGMENT_MAP_URL;

const FEATURE_DEFS: { key: string; label: string }[] = [
  { key: "bearing_change_max", label: "Bearing" },
  { key: "total_crashes", label: "Crashes" },
  { key: "segment_curvature", label: "Curvature" },
  { key: "intersection_degree_max", label: "Intersect" },
  { key: "lanes", label: "Lanes" },
  { key: "avg_predicted_risk", label: "Pred Risk" },
  { key: "maxspeed", label: "Speed" },
  { key: "visibility_risk_score", label: "Visibility" },
];

const CLUSTER_ORDER = [
  "Low-Risk Baseline",
  "Elevated Risk Corridors",
  "Intermediate Risk Group",
  "Moderate-Risk Segments",
  "High-Risk Persistent Segments"
];

function mean(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function median(values: number[]) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }

  return sorted[mid];
}

function arrayMin(values: number[]) {
  if (!values.length) return 0;
  let min = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] < min) min = values[i];
  }
  return min;
}

function arrayMax(values: number[]) {
  if (!values.length) return 0;
  let max = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > max) max = values[i];
  }
  return max;
}

function normalizeByColumn(clusters: ClusterSummary[]) {
  const mins: Record<string, number> = {};
  const maxs: Record<string, number> = {};

  for (const { key } of FEATURE_DEFS) {
    const values = clusters.map((c) => c.values[key] ?? 0);
    mins[key] = arrayMin(values);
    maxs[key] = arrayMax(values);
  }

  return clusters.map((cluster) => {
    const normalized: Record<string, number> = {};

    for (const { key } of FEATURE_DEFS) {
      const min = mins[key];
      const max = maxs[key];
      const value = cluster.values[key] ?? 0;
      normalized[key] = max > min ? (value - min) / (max - min) : 0;
    }

    return {
      ...cluster,
      values: normalized,
    };
  });
}

function colorForValue(value: number) {
  if (value >= 0.85) return "bg-violet-900 text-white";
  if (value >= 0.65) return "bg-violet-700 text-white";
  if (value >= 0.45) return "bg-violet-500 text-white";
  if (value >= 0.25) return "bg-violet-300 text-slate-900";
  if (value >= 0.1) return "bg-violet-100 text-slate-800";
  return "bg-slate-100 text-slate-600";
}

function shortClusterLabel(label: string) {
  switch (label) {
    case "High-Risk Persistent Segments":
      return "Persistent Risk";
    case "Elevated Risk Corridors":
      return "Corridors";
    case "Moderate-Risk Segments":
      return "Moderate Risk";
    case "Intermediate Risk Group":
      return "Intermediate";
    case "Low-Risk Baseline":
      return "Baseline";
    case "Unassigned":
      return "Unassigned";
    default:
      return label;
  }
}

function normalizeClusterLabel(label?: string) {
  const cleaned = label?.trim();
  return cleaned && cleaned.length > 0 ? cleaned : "Unassigned";
}

function bandLabel(speed: number) {
  if (speed <= 20) return "20";
  if (speed <= 25) return "25";
  if (speed <= 30) return "30";
  if (speed <= 40) return "40";
  return "50+";
}

function fmtPct(value: number) {
  return `${(value * 100).toFixed(0)}%`;
}

export default function ClusterFeatureHeatmap() {
  const [data, setData] = useState<GeoJsonCollection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"heatmap" | "speedBands">("heatmap");

  const configError = !MAP_DATA_URL
    ? "NEXT_PUBLIC_SEGMENT_MAP_URL is not defined. Set it in Vercel and .env.local."
    : null;

  useEffect(() => {
    if (!MAP_DATA_URL) return;

    let cancelled = false;

    fetch(MAP_DATA_URL)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load comparison data");
        return res.json();
      })
      .then((json) => {
        if (!cancelled) setData(json as GeoJsonCollection);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const normalizedClusters = useMemo(() => {
    if (!data) return [];

    const grouped = new Map<string, GeoJsonFeature[]>();

    for (const feature of data.features) {
      const label = normalizeClusterLabel(feature.properties.cluster_label);
      if (!grouped.has(label)) grouped.set(label, []);
      grouped.get(label)!.push(feature);
    }

    const summaries: ClusterSummary[] = [];

    for (const label of CLUSTER_ORDER) {
      const features = grouped.get(label);
      if (!features || features.length === 0) continue;

      const props = features.map((f) => f.properties);

      summaries.push({
        cluster_label: label,
        values: {
          bearing_change_max: mean(
            props.map((p) => Number(p.bearing_change_max ?? 0))
          ),
          total_crashes: mean(props.map((p) => Number(p.total_crashes ?? 0))),
          segment_curvature: mean(
            props.map((p) => Number(p.segment_curvature ?? 0))
          ),
          intersection_degree_max: mean(
            props.map((p) => Number(p.intersection_degree_max ?? 0))
          ),
          lanes: mean(props.map((p) => Number(p.lanes ?? 0))),
          avg_predicted_risk: mean(
            props.map((p) => Number(p.avg_predicted_risk ?? 0))
          ),
          maxspeed: mean(props.map((p) => Number(p.maxspeed ?? 0))),
          visibility_risk_score: mean(
            props.map((p) => Number(p.visibility_risk_score ?? 0))
          ),
        },
      });
    }

    return normalizeByColumn(summaries);
  }, [data]);

  const speedBands = useMemo<SpeedBandSummary[]>(() => {
    if (!data) return [];

    const buckets = new Map<string, number[]>();

    for (const feature of data.features) {
      const speed = Number(feature.properties.maxspeed ?? NaN);
      const risk = Number(feature.properties.risk_percentile ?? NaN);

      if (Number.isNaN(speed) || Number.isNaN(risk)) continue;

      const label = bandLabel(speed);
      if (!buckets.has(label)) buckets.set(label, []);
      buckets.get(label)!.push(risk);
    }

    const orderedLabels = ["20", "25", "30", "40", "50+"];

    return orderedLabels
      .filter((label) => buckets.has(label))
      .map((label) => {
        const values = buckets.get(label)!;
        return {
          label,
          values,
          count: values.length,
          mean: mean(values),
          median: median(values),
          min: arrayMin(values),
          max: arrayMax(values),
        };
      });
  }, [data]);

  const displayError = configError || error;

  if (displayError) {
    return <div className="mt-6 text-sm text-red-600">{displayError}</div>;
  }

  if (!normalizedClusters.length) {
    return <div className="mt-6 text-sm text-slate-500">Loading visual comparison...</div>;
  }

  return (
    <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-6">
      <div className="mb-4">
        <h3 className="text-lg font-semibold">Pattern Comparison</h3>
        <p className="mt-1 text-sm text-slate-600">
          Compare cluster-level feature patterns and see how modeled risk changes
          across speed environments.
        </p>
      </div>

      <div className="mb-5 flex gap-2">
        <button
          type="button"
          onClick={() => setActiveTab("heatmap")}
          className={`rounded-full px-4 py-2 text-sm font-medium ${
            activeTab === "heatmap"
              ? "bg-slate-900 text-white"
              : "bg-slate-100 text-slate-700"
          }`}
        >
          Cluster Heatmap
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("speedBands")}
          className={`rounded-full px-4 py-2 text-sm font-medium ${
            activeTab === "speedBands"
              ? "bg-slate-900 text-white"
              : "bg-slate-100 text-slate-700"
          }`}
        >
          Risk by Speed Band
        </button>
      </div>

      {activeTab === "heatmap" ? (
        <div>
          <div className="grid gap-2">
            <div
              className="grid gap-2"
              style={{
                gridTemplateColumns: `170px repeat(${FEATURE_DEFS.length}, minmax(64px, 1fr))`,
              }}
            >
              <div />
              {FEATURE_DEFS.map((feature) => (
                <div
                  key={feature.key}
                  className="px-1 pb-2 text-center text-[11px] font-medium uppercase tracking-[0.06em] text-slate-500"
                >
                  {feature.label}
                </div>
              ))}

              {normalizedClusters.map((cluster) => (
                <div key={cluster.cluster_label} className="contents">
                  <div className="flex items-center rounded-xl border border-slate-200 bg-slate-50 px-3 py-3 text-sm font-medium text-slate-800">
                    {shortClusterLabel(cluster.cluster_label)}
                  </div>

                  {FEATURE_DEFS.map((feature) => {
                    const value = cluster.values[feature.key] ?? 0;

                    return (
                      <div
                        key={`${cluster.cluster_label}-${feature.key}`}
                        className={`flex h-[48px] items-center justify-center rounded-xl text-sm font-semibold ${colorForValue(
                          value
                        )}`}
                        title={`${feature.label}: ${(value * 100).toFixed(0)} normalized`}
                      >
                        {(value * 100).toFixed(0)}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 flex items-center gap-3 text-xs text-slate-600">
            <span>Lower</span>
            <div className="flex h-3 w-40 overflow-hidden rounded">
              <div className="flex-1 bg-slate-100" />
              <div className="flex-1 bg-violet-100" />
              <div className="flex-1 bg-violet-300" />
              <div className="flex-1 bg-violet-500" />
              <div className="flex-1 bg-violet-700" />
              <div className="flex-1 bg-violet-900" />
            </div>
            <span>Higher</span>
          </div>
        </div>
      ) : (
        <div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
            <div className="mb-4 text-sm text-slate-600">
              This view groups segments into speed bands and compares their modeled
              risk percentiles. It is easier to interpret than a raw scatter because
              posted speeds are discrete and many segments share the same speed value.
            </div>

            <div className="relative rounded-2xl bg-slate-800 px-6 py-10">
              <div className="absolute left-0 right-0 top-1/2 h-16 -translate-y-1/2 bg-slate-700" />
              <div className="absolute left-0 right-0 top-1/2 h-1 -translate-y-1/2 border-t-4 border-dashed border-slate-300" />

              <div className="relative grid gap-6 md:grid-cols-5">
                {speedBands.map((band) => (
                  <div key={band.label} className="flex flex-col items-center">
                    <div className="flex h-24 w-24 items-center justify-center rounded-full border-[10px] border-red-600 bg-white text-3xl font-bold text-slate-900 shadow-md">
                      {band.label}
                    </div>

                    <div className="mt-5 flex h-40 w-12 items-end rounded-full bg-slate-200 p-1 shadow-inner">
                      <div
                        className="w-full rounded-full bg-violet-600"
                        style={{ height: `${Math.max(8, band.mean * 100)}%` }}
                        title={`Average risk percentile: ${fmtPct(band.mean)}`}
                      />
                    </div>

                    <div className="mt-4 space-y-1 text-center text-sm">
                      <div className="font-semibold text-white">
                        Avg: {fmtPct(band.mean)}
                      </div>
                      <div className="text-slate-200">
                        Median: {fmtPct(band.median)}
                      </div>
                      <div className="text-slate-300">
                        Range: {fmtPct(band.min)}–{fmtPct(band.max)}
                      </div>
                      <div className="text-slate-400">
                        n = {band.count.toLocaleString()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-5 rounded-2xl border border-slate-200 bg-white p-4">
              <div className="text-xs font-medium uppercase tracking-[0.12em] text-slate-500">
                How to interpret this
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                Each speed-limit sign represents a posted speed band, and the purple
                bar underneath shows the average modeled risk percentile for segments
                in that band. If higher speed bands also show higher average or median
                risk, that suggests speed environment is part of the broader risk story
                — even though other factors like curvature, visibility, and intersection
                complexity still matter.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}