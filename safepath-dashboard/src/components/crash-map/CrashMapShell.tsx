"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import CrashMapControls from "@/src/components/crash-map/CrashMapControls";
import SegmentDetailsPanel from "@/src/components/crash-map/SegmentDetailsPanel";
import {
  InfrastructureOverlay,
  MapMode,
  SelectedSegment,
} from "@/src/lib/crashMap/types";

const CrashMapCanvas = dynamic(
  () => import("@/src/components/crash-map/CrashMapCanvas"),
  { ssr: false }
);

export default function CrashMapShell() {
  const [mode, setMode] = useState<MapMode>("historical");
  const [selectedSegment, setSelectedSegment] = useState<SelectedSegment>(null);
  const [topK, setTopK] = useState<"all" | "top10" | "top5" | "top1">("all");
  const [clusterFilter, setClusterFilter] = useState<string>("all");
  const [overlay, setOverlay] =
    useState<InfrastructureOverlay>("visibility_risk_score");

  const pageDescription = useMemo(() => {
    if (mode === "historical") {
      return "Explore where crashes have historically been concentrated across the street network.";
    }
    if (mode === "predicted") {
      return "View model-estimated crash risk and focus on the highest-priority segments.";
    }
    return "Inspect structural and infrastructural patterns that appear in higher-risk segments.";
  }, [mode]);

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-[1600px] px-6 py-8">
        <div className="mb-6 rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
          <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500">
            Crash Map
          </p>
          <h1 className="mt-3 text-4xl font-bold tracking-tight">
            Explore Crash Risk Across NYC
          </h1>
          <p className="mt-4 max-w-4xl text-slate-600">{pageDescription}</p>
        </div>

        <div className="mb-6">
          <CrashMapControls
            mode={mode}
            onModeChange={setMode}
            topK={topK}
            onTopKChange={setTopK}
            clusterFilter={clusterFilter}
            onClusterFilterChange={setClusterFilter}
            overlay={overlay}
            onOverlayChange={setOverlay}
          />
        </div>

        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
            <CrashMapCanvas
              mode={mode}
              topK={topK}
              clusterFilter={clusterFilter}
              overlay={overlay}
              selectedSegment={selectedSegment}
              onSelectSegment={setSelectedSegment}
            />
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <SegmentDetailsPanel
              mode={mode}
              selectedSegment={selectedSegment}
            />
          </div>
        </div>
      </div>
    </main>
  );
}