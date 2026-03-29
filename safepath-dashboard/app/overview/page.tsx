"use client";

import { useEffect, useState } from "react";
import { loadMetrics } from "@/src/lib/data";

type Metrics = Awaited<ReturnType<typeof loadMetrics>>;

function pct(x: number) {
  return `${(x * 100).toFixed(2)}%`;
}

export default function OverviewPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadMetrics()
      .then(setMetrics)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <main className="min-h-screen bg-white text-slate-900">
      <div className="mx-auto max-w-6xl px-6 py-16">
        <h1 className="text-3xl font-bold">Overview</h1>
        <p className="mt-2 text-slate-600">
          Summary of SafePath’s full-panel crash risk modeling results.
        </p>

        {error && (
          <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">
            {error}
          </div>
        )}

        {!metrics && !error && (
          <p className="mt-6 text-slate-500">Loading metrics...</p>
        )}

        {metrics && (
          <div className="mt-8 grid gap-6 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-2xl border border-slate-200 p-6 shadow-sm">
              <div className="text-sm text-slate-500">Test ROC-AUC</div>
              <div className="mt-2 text-3xl font-semibold">
                {metrics.test_metrics.roc_auc.toFixed(3)}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 p-6 shadow-sm">
              <div className="text-sm text-slate-500">Average Precision</div>
              <div className="mt-2 text-3xl font-semibold">
                {metrics.test_metrics.average_precision.toFixed(3)}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 p-6 shadow-sm">
              <div className="text-sm text-slate-500">Precision @ Top 1%</div>
              <div className="mt-2 text-3xl font-semibold">
                {pct(metrics.test_top_1pct.precision_at_k)}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 p-6 shadow-sm">
              <div className="text-sm text-slate-500">Recall @ Top 5%</div>
              <div className="mt-2 text-3xl font-semibold">
                {pct(metrics.test_top_5pct.recall_at_k)}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}