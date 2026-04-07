"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

type HMMRow = {
  date: string;
  total_crashes: number;
  crash_rate: number;
  avg_crashes_last_7_days: number;
  avg_crashes_last_30_days: number;
  temperature_2m_mean: number;
  precipitation_sum: number;
  windspeed_10m_max: number;
  rain_indicator: number;
  hidden_state: number;
  state_probability_max: number;
  state_label: string;
};

type RegimeSummary = {
  state_label: string;
  n_days: number;
  avg_total_crashes: number;
  avg_crash_rate: number;
  avg_crashes_last_30_days: number;
  avg_precipitation: number;
  avg_temperature: number;
};

function regimeOrder(label: string) {
  if (label.toLowerCase().includes("high")) return 0;
  if (label.toLowerCase().includes("moderate")) return 1;
  if (label.toLowerCase().includes("low")) return 2;
  return 99;
}

export default function HMMRegimeSummary() {
  const [data, setData] = useState<RegimeSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/hmm_daily_states.csv")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load HMM regime data");
        return res.text();
      })
      .then((text) => {
        const lines = text.trim().split("\n");
        const headers = lines[0].split(",");

        const idx = {
          total_crashes: headers.indexOf("total_crashes"),
          crash_rate: headers.indexOf("crash_rate"),
          avg_crashes_last_30_days: headers.indexOf("avg_crashes_last_30_days"),
          precipitation_sum: headers.indexOf("precipitation_sum"),
          temperature_2m_mean: headers.indexOf("temperature_2m_mean"),
          state_label: headers.indexOf("state_label"),
        };

        const required = Object.entries(idx).filter(([, v]) => v === -1);
        if (required.length > 0) {
          throw new Error(
            `Missing required columns in hmm_daily_states.csv: ${required
              .map(([k]) => k)
              .join(", ")}`
          );
        }

        const rows: HMMRow[] = lines.slice(1).map((line) => {
          const cols = line.split(",");
          return {
            date: "",
            total_crashes: Number(cols[idx.total_crashes]),
            crash_rate: Number(cols[idx.crash_rate]),
            avg_crashes_last_7_days: 0,
            avg_crashes_last_30_days: Number(cols[idx.avg_crashes_last_30_days]),
            temperature_2m_mean: Number(cols[idx.temperature_2m_mean]),
            precipitation_sum: Number(cols[idx.precipitation_sum]),
            windspeed_10m_max: 0,
            rain_indicator: 0,
            hidden_state: 0,
            state_probability_max: 0,
            state_label: cols[idx.state_label],
          };
        });

        const grouped = new Map<string, HMMRow[]>();
        for (const row of rows) {
          const arr = grouped.get(row.state_label) ?? [];
          arr.push(row);
          grouped.set(row.state_label, arr);
        }

        const summary: RegimeSummary[] = Array.from(grouped.entries()).map(
          ([label, regimeRows]) => {
            const n = regimeRows.length;

            const avg = (fn: (r: HMMRow) => number) =>
              regimeRows.reduce((sum, r) => sum + fn(r), 0) / n;

            return {
              state_label: label,
              n_days: n,
              avg_total_crashes: avg((r) => r.total_crashes),
              avg_crash_rate: avg((r) => r.crash_rate),
              avg_crashes_last_30_days: avg((r) => r.avg_crashes_last_30_days),
              avg_precipitation: avg((r) => r.precipitation_sum),
              avg_temperature: avg((r) => r.temperature_2m_mean),
            };
          }
        );

        const ordered = summary.sort(
          (a, b) => regimeOrder(a.state_label) - regimeOrder(b.state_label)
        );

        setData(ordered);
      })
      .catch((err) => setError(err.message));
  }, []);

  const cards = useMemo(() => data, [data]);

  if (error) {
    return <div className="text-sm text-red-600">{error}</div>;
  }

  if (!data.length) {
    return <div className="text-sm text-slate-500">Loading regime summary...</div>;
  }

  return (
    <div className="space-y-6">
      <p className="text-xs text-slate-500">
        Citywide HMM regimes summarize recurring periods of lower and higher
        crash activity across the study window.
      </p>

      <div className="h-[280px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 10, right: 20, left: 10, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="state_label" tick={{ fontSize: 12 }} />
            <YAxis tickFormatter={(v) => Number(v).toFixed(3)} />
            <Tooltip
              formatter={(value: number, name: string) => {
                if (name === "avg_crash_rate") return value.toFixed(4);
                return String(value);
              }}
            />
            <Bar dataKey="avg_crash_rate" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {cards.map((regime) => (
          <div
            key={regime.state_label}
            className="rounded-2xl border border-slate-200 bg-white p-5"
          >
            <h4 className="text-base font-semibold text-slate-900">
              {regime.state_label}
            </h4>

            <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-500">Days</div>
                <div className="font-medium">{regime.n_days.toLocaleString()}</div>
              </div>

              <div>
                <div className="text-slate-500">Avg crash rate</div>
                <div className="font-medium">
                  {regime.avg_crash_rate.toFixed(4)}
                </div>
              </div>

              <div>
                <div className="text-slate-500">Avg total crashes</div>
                <div className="font-medium">
                  {regime.avg_total_crashes.toFixed(1)}
                </div>
              </div>

              <div>
                <div className="text-slate-500">Avg 30-day lag</div>
                <div className="font-medium">
                  {regime.avg_crashes_last_30_days.toFixed(3)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}