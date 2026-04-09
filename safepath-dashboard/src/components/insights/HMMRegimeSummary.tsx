"use client";

import { useEffect, useMemo, useState } from "react";

type SummaryRow = {
  hidden_state: number;
  n_days: number;
  avg_total_crashes: number;
  avg_crash_rate: number;
  avg_crashes_last_7_days: number;
  avg_crashes_last_30_days: number;
  avg_temperature_2m_mean: number;
  avg_precipitation_sum: number;
  avg_windspeed_10m_max: number;
  avg_rain_indicator: number;
  state_label: string;
};

type DailyRow = {
  date: string;
  hidden_state: number;
  state_label: string;
};

function parseCsv(text: string) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");

  return lines.slice(1).map((line) => {
    const parts = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = parts[i];
    });
    return row;
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

export default function HMMRegimeSummary() {
  const [summaryRows, setSummaryRows] = useState<SummaryRow[]>([]);
  const [dailyRows, setDailyRows] = useState<DailyRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/data/hmm_state_summary_with_infra.csv").then((r) => {
        if (!r.ok) throw new Error("Failed to load HMM summary");
        return r.text();
      }),
      fetch("/data/hmm_daily_states_with_infra.csv").then((r) => {
        if (!r.ok) throw new Error("Failed to load HMM states");
        return r.text();
      }),
    ])
      .then(([summaryText, statesText]) => {
        const summaryParsed = parseCsv(summaryText).map((row) => ({
          hidden_state: Number(row.hidden_state),
          n_days: Number(row.n_days),
          avg_total_crashes: Number(row.avg_total_crashes),
          avg_crash_rate: Number(row.avg_crash_rate),
          avg_crashes_last_7_days: Number(row.avg_crashes_last_7_days),
          avg_crashes_last_30_days: Number(row.avg_crashes_last_30_days),
          avg_temperature_2m_mean: Number(row.avg_temperature_2m_mean),
          avg_precipitation_sum: Number(row.avg_precipitation_sum),
          avg_windspeed_10m_max: Number(row.avg_windspeed_10m_max),
          avg_rain_indicator: Number(row.avg_rain_indicator),
          state_label: row.state_label,
        }));

        const statesParsed = parseCsv(statesText).map((row) => ({
          date: row.date,
          hidden_state: Number(row.hidden_state),
          state_label: row.state_label,
        }));

        setSummaryRows(summaryParsed);
        setDailyRows(statesParsed);
      })
      .catch((err) => setError(err.message));
  }, []);

  const transitionCounts = useMemo(() => {
    if (!dailyRows.length) return [];

    const ordered = [...dailyRows].sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    const labels = Array.from(new Set(ordered.map((r) => r.state_label)));
    const counts: Record<string, Record<string, number>> = {};

    labels.forEach((from) => {
      counts[from] = {};
      labels.forEach((to) => {
        counts[from][to] = 0;
      });
    });

    for (let i = 0; i < ordered.length - 1; i++) {
      const from = ordered[i].state_label;
      const to = ordered[i + 1].state_label;
      counts[from][to] += 1;
    }

    return labels.map((from) => ({
      from,
      ...counts[from],
    }));
  }, [dailyRows]);

  const sortedSummary = useMemo(() => {
    return [...summaryRows].sort(
      (a, b) => b.avg_total_crashes - a.avg_total_crashes
    );
  }, [summaryRows]);

  if (error) {
    return <div className="text-sm text-red-600">{error}</div>;
  }

  if (!sortedSummary.length) {
    return <div className="text-sm text-slate-500">Loading HMM summary...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-3">
        {sortedSummary.map((state) => (
          <div
            key={state.hidden_state}
            className="rounded-2xl border border-slate-200 bg-slate-50 p-6"
          >
            <h3 className="text-lg font-semibold">{state.state_label}</h3>
            <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-500">Days</div>
                <div className="font-medium">{state.n_days.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-slate-500">Avg crashes</div>
                <div className="font-medium">{fmt(state.avg_total_crashes, 1)}</div>
              </div>
              <div>
                <div className="text-slate-500">Avg crash rate</div>
                <div className="font-medium">{fmt(state.avg_crash_rate, 4)}</div>
              </div>
              <div>
                <div className="text-slate-500">Rain days</div>
                <div className="font-medium">{pct(state.avg_rain_indicator)}</div>
              </div>
              <div>
                <div className="text-slate-500">Avg precipitation</div>
                <div className="font-medium">
                  {fmt(state.avg_precipitation_sum, 2)}
                </div>
              </div>
              <div>
                <div className="text-slate-500">Avg wind</div>
                <div className="font-medium">
                  {fmt(state.avg_windspeed_10m_max, 1)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6">
        <h3 className="text-lg font-semibold">Transition behavior</h3>
        <p className="mt-2 text-sm text-slate-600">
          Counts below show how often the city moved between daily regimes from
          one day to the next.
        </p>

        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full border-separate border-spacing-0 text-sm">
            <thead>
              <tr className="text-left text-slate-500">
                <th className="border-b border-slate-200 px-3 py-2 font-medium">
                  From state
                </th>
                {sortedSummary.map((s) => (
                  <th
                    key={s.state_label}
                    className="border-b border-slate-200 px-3 py-2 font-medium"
                  >
                    To {s.state_label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {transitionCounts.map((row) => (
                <tr key={row.from}>
                  <td className="border-b border-slate-100 px-3 py-2 font-medium">
                    {row.from}
                  </td>
                  {sortedSummary.map((s) => (
                    <td
                      key={`${row.from}-${s.state_label}`}
                      className="border-b border-slate-100 px-3 py-2"
                    >
                      {row[s.state_label as keyof typeof row] as unknown as number}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <p className="text-sm leading-6 text-slate-600">
            The high-risk regime represents event-like periods with much higher
            average crash counts than the city’s normal operating baseline.
          </p>
          <p className="text-sm leading-6 text-slate-600">
            The transition structure suggests that once the city enters its
            elevated regime, those conditions tend to persist rather than
            disappearing immediately the next day.
          </p>
        </div>
      </div>
    </div>
  );
}