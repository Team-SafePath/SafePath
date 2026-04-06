"use client";

import { useEffect, useMemo, useState } from "react";

type ClusterRow = {
  gmm_cluster: number;
  avg_crash_rate: number;
  total_crashes: number;
  avg_crashes_last_30_days: number;
};

type ClusterSummary = {
  gmm_cluster: number;
  cluster_label: string;
  avg_crash_rate: number;
  avg_total_crashes: number;
  avg_crashes_last_30_days: number;
  n_segments: number;
};

function getClusterLabel(rank: number, size: number) {
  if (rank === 0) return "Persistent Crash Zones";
  if (rank === 1) return "Elevated Risk Corridors";
  if (rank === 2) return "Moderate-Risk Segments";
  if (rank === size - 1) return "Low-Risk Baseline";
  return "Intermediate Risk Group";
}

export default function GMMClusterStats() {
  const [data, setData] = useState<ClusterSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/segment_clusters.csv")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load GMM cluster data");
        return res.text();
      })
      .then((text) => {
        const lines = text.trim().split("\n");
        const headers = lines[0].split(",");

        const idx = {
          gmm_cluster: headers.indexOf("gmm_cluster"),
          avg_crash_rate: headers.indexOf("avg_crash_rate"),
          total_crashes: headers.indexOf("total_crashes"),
          avg_crashes_last_30_days: headers.indexOf("avg_crashes_last_30_days"),
        };

        const required = Object.entries(idx).filter(([, v]) => v === -1);
        if (required.length > 0) {
          throw new Error(
            `Missing required columns in segment_clusters.csv: ${required
              .map(([k]) => k)
              .join(", ")}`
          );
        }

        const rows: ClusterRow[] = lines.slice(1).map((line) => {
          const cols = line.split(",");
          return {
            gmm_cluster: Number(cols[idx.gmm_cluster]),
            avg_crash_rate: Number(cols[idx.avg_crash_rate]),
            total_crashes: Number(cols[idx.total_crashes]),
            avg_crashes_last_30_days: Number(cols[idx.avg_crashes_last_30_days]),
          };
        });

        const grouped = new Map<number, ClusterRow[]>();
        for (const row of rows) {
          const arr = grouped.get(row.gmm_cluster) ?? [];
          arr.push(row);
          grouped.set(row.gmm_cluster, arr);
        }

        const summary: ClusterSummary[] = Array.from(grouped.entries()).map(
          ([cluster, clusterRows]) => {
            const n = clusterRows.length;
            const avg = (fn: (r: ClusterRow) => number) =>
              clusterRows.reduce((sum, r) => sum + fn(r), 0) / n;

            return {
              gmm_cluster: cluster,
              cluster_label: "",
              avg_crash_rate: avg((r) => r.avg_crash_rate),
              avg_total_crashes: avg((r) => r.total_crashes),
              avg_crashes_last_30_days: avg((r) => r.avg_crashes_last_30_days),
              n_segments: n,
            };
          }
        );

        const ordered = [...summary].sort(
          (a, b) => b.avg_crash_rate - a.avg_crash_rate
        );

        const withLabels = ordered.map((row, i) => ({
          ...row,
          cluster_label: getClusterLabel(i, ordered.length),
        }));

        setData(withLabels);
      })
      .catch((err) => setError(err.message));
  }, []);

  const cards = useMemo(() => data.slice(0, 4), [data]);

  if (error) {
    return <div className="text-sm text-red-600">{error}</div>;
  }

  if (!data.length) {
    return <div className="text-sm text-slate-500">Loading cluster statistics...</div>;
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {cards.map((cluster) => (
        <div
          key={cluster.gmm_cluster}
          className="rounded-2xl border border-slate-200 bg-white p-5"
        >
          <h4 className="text-base font-semibold text-slate-900">
            {cluster.cluster_label}
          </h4>

          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-slate-500">Segments</div>
              <div className="font-medium">{cluster.n_segments.toLocaleString()}</div>
            </div>

            <div>
              <div className="text-slate-500">Avg crash rate</div>
              <div className="font-medium">
                {cluster.avg_crash_rate.toFixed(4)}
              </div>
            </div>

            <div>
              <div className="text-slate-500">Avg total crashes</div>
              <div className="font-medium">
                {cluster.avg_total_crashes.toFixed(1)}
              </div>
            </div>

            <div>
              <div className="text-slate-500">Avg 30-day lag</div>
              <div className="font-medium">
                {cluster.avg_crashes_last_30_days.toFixed(3)}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}