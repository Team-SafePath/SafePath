"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

type ClusterRow = {
  segment_id: number;
  gmm_cluster: number;
  avg_crash_rate: number;
  total_crashes: number;
  avg_crashes_last_30_days: number;
};

type ClusterSummary = {
  gmm_cluster: number;
  cluster_label: string;
  avg_crash_rate: number;
};

function getClusterLabel(rank: number, size: number) {
  if (rank === 0) return "Persistent Crash Zones";
  if (rank === 1) return "Elevated Risk Corridors";
  if (rank === 2) return "Moderate-Risk Segments";
  if (rank === size - 1) return "Low-Risk Baseline";
  return "Intermediate Risk Group";
}

export default function GMMClusterChart() {
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
            segment_id: 0,
            gmm_cluster: Number(cols[idx.gmm_cluster]),
            avg_crash_rate: Number(cols[idx.avg_crash_rate]),
            total_crashes: 0,
            avg_crashes_last_30_days: 0,
          };
        });

        const grouped = new Map<number, number[]>();
        for (const row of rows) {
          const arr = grouped.get(row.gmm_cluster) ?? [];
          arr.push(row.avg_crash_rate);
          grouped.set(row.gmm_cluster, arr);
        }

        const summary: ClusterSummary[] = Array.from(grouped.entries()).map(
          ([cluster, values]) => ({
            gmm_cluster: cluster,
            cluster_label: "",
            avg_crash_rate:
              values.reduce((sum, v) => sum + v, 0) / values.length,
          })
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

  if (error) {
    return <div className="text-sm text-red-600">{error}</div>;
  }

  if (!data.length) {
    return <div className="text-sm text-slate-500">Loading cluster chart...</div>;
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-slate-500">
        Average crash rate by unsupervised segment archetype.
      </p>

      <div className="w-full">
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={data}
            margin={{ top: 10, right: 10, left: 10, bottom: 70 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="cluster_label"
              angle={-15}
              textAnchor="end"
              interval={0}
              height={70}
              tick={{ fontSize: 11 }}
            />
            <YAxis tickFormatter={(v) => Number(v).toFixed(3)} />
            <Tooltip
              formatter={(value) => {
                const numericValue =
                  typeof value === "number" ? value : Number(value ?? 0);
                return numericValue.toFixed(4);
              }}
            />
            <Bar dataKey="avg_crash_rate" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}