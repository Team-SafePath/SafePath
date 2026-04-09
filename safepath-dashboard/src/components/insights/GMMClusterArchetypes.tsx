"use client";

import { useEffect, useMemo, useState } from "react";

type ClusterRow = {
  gmm_cluster: number;
  n_segments: number;
  total_crashes?: number;
  avg_crash_rate?: number;
  avg_crash_volatility?: number;
  avg_pct_days_with_crash?: number;
  avg_avg_crashes_last_7_days?: number;
  avg_avg_crashes_last_30_days?: number;
  avg_segment_length?: number;
  avg_lanes?: number;
  avg_maxspeed?: number;
  avg_segment_curvature?: number;
  avg_bearing_change_max?: number;
  avg_intersection_degree_max?: number;
  avg_visibility_risk_score?: number;
  avg_near_intersection?: number;
  avg_near_traffic_signal?: number;
  cluster_label: string;
  dominant_road_type?: string;
};

function parseCsv(text: string): ClusterRow[] {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");

  return lines.slice(1).map((line) => {
    const parts = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = parts[i];
    });

    return {
      gmm_cluster: Number(row.gmm_cluster),
      n_segments: Number(row.n_segments),
      total_crashes: Number(row.total_crashes),
      avg_crash_rate: Number(row.avg_crash_rate),
      avg_crash_volatility: Number(row.avg_crash_volatility),
      avg_pct_days_with_crash: Number(row.avg_pct_days_with_crash),
      avg_avg_crashes_last_7_days: Number(row.avg_avg_crashes_last_7_days),
      avg_avg_crashes_last_30_days: Number(row.avg_avg_crashes_last_30_days),
      avg_segment_length: Number(row.avg_segment_length),
      avg_lanes: Number(row.avg_lanes),
      avg_maxspeed: Number(row.avg_maxspeed),
      avg_segment_curvature: Number(row.avg_segment_curvature),
      avg_bearing_change_max: Number(row.avg_bearing_change_max),
      avg_intersection_degree_max: Number(row.avg_intersection_degree_max),
      avg_visibility_risk_score: Number(row.avg_visibility_risk_score),
      avg_near_intersection: Number(row.avg_near_intersection),
      avg_near_traffic_signal: Number(row.avg_near_traffic_signal),
      cluster_label: row.cluster_label,
      dominant_road_type: row.dominant_road_type,
    };
  });
}

function fmt(value: number | undefined, digits = 2) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

function pct(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(0)}%`;
}

export default function GMMClusterArchetypes() {
  const [rows, setRows] = useState<ClusterRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/gmm_cluster_summary_with_infra.csv")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load GMM summary");
        return res.text();
      })
      .then((text) => setRows(parseCsv(text)))
      .catch((err) => setError(err.message));
  }, []);

  const sorted = useMemo(() => {
    return [...rows].sort(
      (a, b) => (b.avg_crash_rate ?? 0) - (a.avg_crash_rate ?? 0)
    );
  }, [rows]);

  if (error) {
    return <div className="text-sm text-red-600">{error}</div>;
  }

  if (!sorted.length) {
    return <div className="text-sm text-slate-500">Loading GMM summary...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-2">
        {sorted.map((cluster) => (
          <div
            key={cluster.gmm_cluster}
            className="rounded-2xl border border-slate-200 bg-slate-50 p-6"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold">{cluster.cluster_label}</h3>
                <p className="mt-1 text-sm text-slate-600">
                  Dominant road type: {cluster.dominant_road_type ?? "N/A"}
                </p>
              </div>
              <div className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-600">
                {cluster.n_segments.toLocaleString()} segments
              </div>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-500">Avg crash rate</div>
                <div className="font-medium">{fmt(cluster.avg_crash_rate, 4)}</div>
              </div>
              <div>
                <div className="text-slate-500">Avg total crashes</div>
                <div className="font-medium">{fmt(cluster.total_crashes, 1)}</div>
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
                  {pct(cluster.avg_near_traffic_signal)}
                </div>
              </div>
              <div>
                <div className="text-slate-500">Near intersection</div>
                <div className="font-medium">
                  {pct(cluster.avg_near_intersection)}
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
            The spatial clustering identifies recurring types of segment
            environments rather than only ranking roads one by one. That makes
            the map easier to interpret: some segments belong to a persistent
            residential risk pattern, while others resemble faster corridor-type
            environments.
          </p>
          <p className="text-sm leading-6 text-slate-600">
            These archetypes help explain that risk is not spatially random.
            Different groups combine distinct mixtures of road design,
            intersection context, visibility risk, and historical crash
            behavior.
          </p>
        </div>
      </div>
    </div>
  );
}