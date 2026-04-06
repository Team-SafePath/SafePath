"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

type FeatureRow = {
  feature: string;
  importance: number;
  importance_pct: number;
};

function formatFeatureName(name: string) {
  const replacements: Record<string, string> = {
    temperature_2m_mean: "Temperature",
    precipitation_sum: "Precipitation",
    windspeed_10m_max: "Wind Speed",
    crashes_last_30_days: "Crashes (Last 30 Days)",
    crashes_last_7_days: "Crashes (Last 7 Days)",
    day_of_week: "Day of Week",
    month: "Month",
    sin_day_of_week: "Day of Week (Cyclic)",
    cos_day_of_week: "Day of Week (Cyclic)",
    sin_month: "Month (Cyclic)",
    cos_month: "Month (Cyclic)",
    segment_length: "Segment Length",
    road_residential: "Residential Road",
    road_secondary: "Secondary Road",
    road_primary: "Primary Road",
    road_motorway: "Motorway",
    rain_indicator: "Rain Indicator",
  };

  return replacements[name] ?? name.replaceAll("_", " ");
}

export default function FeatureImportanceChart() {
  const [data, setData] = useState<FeatureRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/lightgbm_full_panel_feature_importance.csv")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load feature importance");
        return res.text();
      })
      .then((text) => {
        const lines = text.trim().split("\n");

        const parsed = lines.slice(1).map((line) => {
          const [feature, importance] = line.split(",");
          return {
            feature,
            importance: Number(importance),
          };
        });

        const sorted = parsed.sort((a, b) => b.importance - a.importance);
        const top = sorted.slice(0, 12);

        const total = top.reduce((sum, d) => sum + d.importance, 0);

        const normalized = top.map((d) => ({
          ...d,
          importance_pct: total > 0 ? (d.importance / total) * 100 : 0,
        }));

        setData(normalized);
      })
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return <div className="text-sm text-red-600">{error}</div>;
  }

  if (!data.length) {
    return <div className="text-sm text-slate-500">Loading chart...</div>;
  }

  return (
    <div className="w-full">
      <p className="mb-3 text-xs text-slate-500">
        Relative contribution of the top model features, normalized across the
        top 12 drivers.
      </p>

      <div className="h-[350px] w-full flex justify-start">
        <div className="w-[90%]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              layout="vertical"
              margin={{ top: 10, right: 20, left: 0, bottom: 10 }}
            >
              <XAxis
                type="number"
                tickFormatter={(v) => `${Number(v).toFixed(0)}%`}
              />
              <YAxis
                type="category"
                dataKey="feature"
                width={170}
                tick={{ fontSize: 12 }}
                tickFormatter={formatFeatureName}
              />
              <Tooltip
                formatter={(value) => {
                    const numericValue =
                    typeof value === "number" ? value : Number(value ?? 0);
                    return `${numericValue.toFixed(1)}%`;
                }}
                labelFormatter={(label) => formatFeatureName(String(label))}
              />
              <Bar dataKey="importance_pct" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}