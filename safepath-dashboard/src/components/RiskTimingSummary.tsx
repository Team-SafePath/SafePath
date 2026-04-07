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
  avg_crashes_last_30_days: number;
  precipitation_sum: number;
  temperature_2m_mean: number;
  state_label: string;
};

type MonthlySummary = {
  month: string;
  high_risk_days: number;
  total_days: number;
  high_risk_share_pct: number;
  avg_crash_rate: number;
};

const MONTH_ORDER = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

function shortMonth(dateStr: string) {
  const d = new Date(dateStr);
  return d.toLocaleString("en-US", { month: "short" });
}

function isHighRisk(label: string) {
  return label.toLowerCase().includes("high");
}

export default function RiskTimingSummary() {
  const [rows, setRows] = useState<HMMRow[]>([]);
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
          date: headers.indexOf("date"),
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

        const parsed: HMMRow[] = lines.slice(1).map((line) => {
          const cols = line.split(",");
          return {
            date: cols[idx.date],
            total_crashes: Number(cols[idx.total_crashes]),
            crash_rate: Number(cols[idx.crash_rate]),
            avg_crashes_last_30_days: Number(cols[idx.avg_crashes_last_30_days]),
            precipitation_sum: Number(cols[idx.precipitation_sum]),
            temperature_2m_mean: Number(cols[idx.temperature_2m_mean]),
            state_label: cols[idx.state_label],
          };
        });

        setRows(parsed);
      })
      .catch((err) => setError(err.message));
  }, []);

  const monthly = useMemo<MonthlySummary[]>(() => {
    if (!rows.length) return [];

    const grouped = new Map<string, HMMRow[]>();

    rows.forEach((row) => {
      const m = shortMonth(row.date);
      const arr = grouped.get(m) ?? [];
      arr.push(row);
      grouped.set(m, arr);
    });

    return MONTH_ORDER.map((month) => {
      const monthRows = grouped.get(month) ?? [];
      const totalDays = monthRows.length;
      const highRiskDays = monthRows.filter((r) => isHighRisk(r.state_label)).length;
      const avgCrashRate =
        totalDays > 0
          ? monthRows.reduce((sum, r) => sum + r.crash_rate, 0) / totalDays
          : 0;

      return {
        month,
        high_risk_days: highRiskDays,
        total_days: totalDays,
        high_risk_share_pct: totalDays > 0 ? (highRiskDays / totalDays) * 100 : 0,
        avg_crash_rate: avgCrashRate,
      };
    });
  }, [rows]);

  const summary = useMemo(() => {
    if (!rows.length || !monthly.length) return null;

    const highRiskRows = rows.filter((r) => isHighRisk(r.state_label));
    const lowRiskRows = rows.filter((r) =>
      r.state_label.toLowerCase().includes("low")
    );

    const avg = (vals: number[]) =>
      vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;

    const peakMonth = [...monthly].sort(
      (a, b) => b.high_risk_share_pct - a.high_risk_share_pct
    )[0];

    const lowMonth = [...monthly].sort(
      (a, b) => a.high_risk_share_pct - b.high_risk_share_pct
    )[0];

    return {
      peakMonth,
      lowMonth,
      highRiskAvgCrashRate: avg(highRiskRows.map((r) => r.crash_rate)),
      lowRiskAvgCrashRate: avg(lowRiskRows.map((r) => r.crash_rate)),
    };
  }, [rows, monthly]);

  if (error) {
    return <div className="text-sm text-red-600">{error}</div>;
  }

  if (!rows.length) {
    return <div className="text-sm text-slate-500">Loading timing summary...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
        <h3 className="text-lg font-semibold">Monthly high-risk pattern</h3>
        <p className="mt-2 text-sm text-slate-600">
          Share of days in each month classified as belonging to the high-risk regime.
        </p>

        <div className="mt-4 h-[280px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={monthly}
              margin={{ top: 10, right: 20, left: 10, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="month" tick={{ fontSize: 12 }} />
              <YAxis tickFormatter={(v) => `${Number(v).toFixed(0)}%`} />
              <Tooltip
                formatter={(value: number) => `${value.toFixed(1)}%`}
              />
              <Bar dataKey="high_risk_share_pct" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {summary && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6">
          <h3 className="text-lg font-semibold">Timing takeaways</h3>
          <p className="mt-2 text-sm text-slate-600">
            Concrete time-based summaries from the HMM regime output.
          </p>

          <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm text-slate-500">Peak high-risk month</div>
              <div className="mt-2 text-2xl font-semibold">
                {summary.peakMonth.month}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                {summary.peakMonth.high_risk_share_pct.toFixed(1)}% of days high-risk
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm text-slate-500">Lowest high-risk month</div>
              <div className="mt-2 text-2xl font-semibold">
                {summary.lowMonth.month}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                {summary.lowMonth.high_risk_share_pct.toFixed(1)}% of days high-risk
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm text-slate-500">Avg crash rate on high-risk days</div>
              <div className="mt-2 text-2xl font-semibold">
                {summary.highRiskAvgCrashRate.toFixed(4)}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                Mean citywide crash rate in the high-risk regime
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm text-slate-500">Avg crash rate on low-risk days</div>
              <div className="mt-2 text-2xl font-semibold">
                {summary.lowRiskAvgCrashRate.toFixed(4)}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                Mean citywide crash rate in the low-risk regime
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}