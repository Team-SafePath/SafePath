"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Row = {
  feature: string;
  importance: number;
};

function parseCsv(text: string): Row[] {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");
  const featureIdx = headers.indexOf("feature");
  const importanceIdx = headers.indexOf("importance");

  if (featureIdx === -1 || importanceIdx === -1) {
    throw new Error("Feature importance CSV is missing required columns.");
  }

  return lines.slice(1).map((line) => {
    const parts = line.split(",");
    return {
      feature: parts[featureIdx],
      importance: Number(parts[importanceIdx]),
    };
  });
}

function formatFeatureName(feature: string) {
  return feature
    .replaceAll("_", " ")
    .replace("visibility risk score", "visibility risk")
    .replace("segment curvature", "curvature")
    .replace("bearing change max", "bearing change")
    .replace("intersection degree max", "intersection degree");
}

export default function FeatureImportanceChart() {
  const [rows, setRows] = useState<Row[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/lightgbm_full_panel_with_infra_feature_importance.csv")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load feature importance CSV");
        return res.text();
      })
      .then((text) => setRows(parseCsv(text)))
      .catch((err) => setError(err.message));
  }, []);

  const topRows = useMemo(() => {
    return rows.slice(0, 12).map((r) => ({
      ...r,
      feature_label: formatFeatureName(r.feature),
    }));
  }, [rows]);

  if (error) {
    return <div className="text-sm text-red-600">{error}</div>;
  }

  if (!topRows.length) {
    return <div className="text-sm text-slate-500">Loading chart...</div>;
  }

  return (
    <div className="space-y-3">
      <div className="h-[420px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={topRows}
            layout="vertical"
            margin={{ top: 10, right: 20, left: 40, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" />
            <YAxis
              type="category"
              dataKey="feature_label"
              width={170}
              tick={{ fontSize: 12 }}
            />
            <Tooltip
              formatter={(value) => {
                const n = typeof value === "number" ? value : Number(value ?? 0);
                return `${n.toFixed(0)}`;
              }}
            />
            <Bar dataKey="importance" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-slate-500">
        Higher importance indicates that the model relied more heavily on that
        feature when ranking crash risk across segments.
      </p>
    </div>
  );
}