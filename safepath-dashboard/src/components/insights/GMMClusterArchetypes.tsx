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
    segment_length?: number;
    lanes?: number;
    maxspeed?: number;
    segment_curvature?: number;
    bearing_change_max?: number;
    intersection_degree_max?: number;
    visibility_risk_score?: number;
    near_intersection?: number;
    near_traffic_signal?: number;
  };
  geometry: GeoJSON.Geometry;
};

type GeoJsonCollection = {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
};

type ClusterSummary = {
  gmm_cluster: number;
  cluster_label: string;
  n_segments: number;
  dominant_road_type: string;
  avg_total_crashes: number;
  avg_predicted_risk: number;
  avg_risk_percentile: number;
  avg_segment_length: number;
  avg_lanes: number;
  avg_maxspeed: number;
  avg_segment_curvature: number;
  avg_bearing_change_max: number;
  avg_intersection_degree_max: number;
  avg_visibility_risk_score: number;
  avg_near_intersection: number;
  avg_near_traffic_signal: number;
};

const MAP_DATA_URL = process.env.NEXT_PUBLIC_SEGMENT_MAP_URL;

function mean(values: number[]) {
  if (!values.length) return NaN;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function fmt(value: number | undefined, digits = 2) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

function pctFromFraction(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(0)}%`;
}

function pctFromPercentile(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

function summarizeClusters(features: GeoJsonFeature[]): ClusterSummary[] {
  const grouped = new Map<number, GeoJsonFeature[]>();

  for (const feature of features) {
    const clusterId = feature.properties.gmm_cluster;
    if (clusterId === undefined || Number.isNaN(clusterId)) continue;

    if (!grouped.has(clusterId)) {
      grouped.set(clusterId, []);
    }
    grouped.get(clusterId)!.push(feature);
  }

  const summaries: ClusterSummary[] = [];

  for (const [clusterId, clusterFeatures] of grouped.entries()) {
    const propsList = clusterFeatures.map((f) => f.properties);

    const roadTypeCounts = new Map<string, number>();
    for (const p of propsList) {
      const roadType = p.road_type ?? "unknown";
      roadTypeCounts.set(roadType, (roadTypeCounts.get(roadType) ?? 0) + 1);
    }

    let dominantRoadType = "N/A";
    let maxCount = -1;
    for (const [roadType, count] of roadTypeCounts.entries()) {
      if (count > maxCount) {
        dominantRoadType = roadType;
        maxCount = count;
      }
    }

    const clusterLabel =
      propsList.find((p) => p.cluster_label)?.cluster_label ??
      `Cluster ${clusterId}`;

    summaries.push({
      gmm_cluster: clusterId,
      cluster_label: clusterLabel,
      n_segments: propsList.length,
      dominant_road_type: dominantRoadType,
      avg_total_crashes: mean(propsList.map((p) => Number(p.total_crashes ?? 0))),
      avg_predicted_risk: mean(
        propsList.map((p) => Number(p.avg_predicted_risk ?? 0))
      ),
      avg_risk_percentile: mean(
        propsList.map((p) => Number(p.risk_percentile ?? 0))
      ),
      avg_segment_length: mean(
        propsList.map((p) => Number(p.segment_length ?? 0))
      ),
      avg_lanes: mean(propsList.map((p) => Number(p.lanes ?? 0))),
      avg_maxspeed: mean(propsList.map((p) => Number(p.maxspeed ?? 0))),
      avg_segment_curvature: mean(
        propsList.map((p) => Number(p.segment_curvature ?? 0))
      ),
      avg_bearing_change_max: mean(
        propsList.map((p) => Number(p.bearing_change_max ?? 0))
      ),
      avg_intersection_degree_max: mean(
        propsList.map((p) => Number(p.intersection_degree_max ?? 0))
      ),
      avg_visibility_risk_score: mean(
        propsList.map((p) => Number(p.visibility_risk_score ?? 0))
      ),
      avg_near_intersection: mean(
        propsList.map((p) => Number(p.near_intersection ?? 0))
      ),
      avg_near_traffic_signal: mean(
        propsList.map((p) => Number(p.near_traffic_signal ?? 0))
      ),
    });
  }

  return summaries.sort((a, b) => b.avg_predicted_risk - a.avg_predicted_risk);
}

export default function GMMClusterArchetypes() {
  const [data, setData] = useState<GeoJsonCollection | null>(null);
  const [error, setError] = useState<string | null>(null);

  const configError = !MAP_DATA_URL
    ? "NEXT_PUBLIC_SEGMENT_MAP_URL is not defined. Set it in Vercel and .env.local."
    : null;

  useEffect(() => {
    if (!MAP_DATA_URL) return;

    let cancelled = false;

    fetch(MAP_DATA_URL)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load cluster map data");
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

  const summaries = useMemo(() => {
    if (!data) return [];
    return summarizeClusters(data.features);
  }, [data]);

  const displayError = configError || error;

  if (displayError) {
    return <div className="text-sm text-red-600">{displayError}</div>;
  }

  if (!summaries.length) {
    return (
      <div className="text-sm text-slate-500">Loading cluster archetypes...</div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-2">
        {summaries.map((cluster) => (
          <div
            key={cluster.gmm_cluster}
            className="rounded-2xl border border-slate-200 bg-slate-50 p-6"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold">{cluster.cluster_label}</h3>
                <p className="mt-1 text-sm text-slate-600">
                  Dominant road type: {cluster.dominant_road_type}
                </p>
              </div>
              <div className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-600">
                {cluster.n_segments.toLocaleString()} segments
              </div>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-500">Avg total crashes</div>
                <div className="font-medium">
                  {fmt(cluster.avg_total_crashes, 1)}
                </div>
              </div>

              <div>
                <div className="text-slate-500">Avg predicted risk</div>
                <div className="font-medium">
                  {fmt(cluster.avg_predicted_risk, 4)}
                </div>
              </div>

              <div>
                <div className="text-slate-500">Avg risk percentile</div>
                <div className="font-medium">
                  {pctFromPercentile(cluster.avg_risk_percentile)}
                </div>
              </div>

              <div>
                <div className="text-slate-500">Avg visibility risk</div>
                <div className="font-medium">
                  {fmt(cluster.avg_visibility_risk_score, 3)}
                </div>
              </div>

              <div>
                <div className="text-slate-500">Avg curvature</div>
                <div className="font-medium">
                  {fmt(cluster.avg_segment_curvature, 3)}
                </div>
              </div>

              <div>
                <div className="text-slate-500">Avg bearing change</div>
                <div className="font-medium">
                  {fmt(cluster.avg_bearing_change_max, 2)}
                </div>
              </div>

              <div>
                <div className="text-slate-500">Avg lanes</div>
                <div className="font-medium">{fmt(cluster.avg_lanes, 1)}</div>
              </div>

              <div>
                <div className="text-slate-500">Avg max speed</div>
                <div className="font-medium">{fmt(cluster.avg_maxspeed, 1)}</div>
              </div>

              <div>
                <div className="text-slate-500">Near traffic signal</div>
                <div className="font-medium">
                  {pctFromFraction(cluster.avg_near_traffic_signal)}
                </div>
              </div>

              <div>
                <div className="text-slate-500">Near intersection</div>
                <div className="font-medium">
                  {pctFromFraction(cluster.avg_near_intersection)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6">
        <h3 className="text-lg font-semibold">Interpretation</h3>
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <p className="text-sm leading-6 text-slate-600">
            These spatial clusters represent recurring roadway archetypes rather
            than isolated segments. That helps explain where elevated risk tends
            to appear: some clusters resemble persistent residential-risk
            environments, while others align more closely with faster corridor
            conditions and more complex geometry.
          </p>
          <p className="text-sm leading-6 text-slate-600">
            In other words, crash risk is not spatially random. The clusters
            show that different groups of segments combine distinct mixtures of
            speed environment, road design, visibility risk, and intersection
            context, creating recognizable patterns across the network.
          </p>
        </div>
      </div>
    </div>
  );
}