"use client";

import dynamic from "next/dynamic";

const CrashMap = dynamic(() => import("@/src/components/CrashMap"), {
  ssr: false,
});

export default function CrashMapPage() {
  return (
    <main className="min-h-screen bg-white text-slate-900">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <div className="mb-8">
          <h1 className="text-3xl font-bold">Crash Map</h1>
          <p className="mt-2 max-w-3xl text-slate-600">
            Compare historical crash concentration with modeled crash risk across
            New York City street segments. Use the toggle to switch between
            historical crash frequency and predicted risk.
          </p>
        </div>

        <CrashMap />
      </div>
    </main>
  );
}