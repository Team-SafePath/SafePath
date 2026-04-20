"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import CrashMapControls from "@/src/components/crash-map/CrashMapControls";
import SegmentDetailsPanel from "@/src/components/crash-map/SegmentDetailsPanel";
import TopRiskSegmentsPanel from "@/src/components/crash-map/TopRiskSegmentsPanel";
import {
  InfrastructureOverlay,
  MapMode,
  SelectedSegment,
} from "@/src/lib/crashMap/types";

const CrashMapCanvas = dynamic(
  () => import("@/src/components/crash-map/CrashMapCanvas"),
  { ssr: false }
);

type SidePanelTab = "details" | "top-risk";

export default function CrashMapShell() {
  const [mode, setMode] = useState<MapMode>("historical");
  const [selectedSegment, setSelectedSegment] = useState<SelectedSegment>(null);
  const [clusterFilter, setClusterFilter] = useState<string>("all");
  const [overlay, setOverlay] =
    useState<InfrastructureOverlay>("visibility_risk_score");
  const [sidePanelTab, setSidePanelTab] = useState<SidePanelTab>("details");

  const pageDescription = useMemo(() => {
    if (mode === "historical") {
      return "Explore where crashes have historically concentrated across the street network and identify segments with the strongest observed crash burden.";
    }

    if (mode === "predicted") {
      return "View model-estimated crash risk across all segments using percentile-based coloring, and use cluster filters to see which roadway archetypes dominate the highest-risk predicted areas.";
    }

    return "Inspect structural roadway patterns using infrastructure overlays such as visibility risk, curvature, speed environment, and intersection complexity.";
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
            clusterFilter={clusterFilter}
            onClusterFilterChange={setClusterFilter}
            overlay={overlay}
            onOverlayChange={setOverlay}
          />
        </div>

        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
            <CrashMapCanvas
              mode={mode}
              clusterFilter={clusterFilter}
              overlay={overlay}
              selectedSegment={selectedSegment}
              onSelectSegment={(segment) => {
                setSelectedSegment(segment);
                setSidePanelTab("details");
              }}
            />
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-5 flex gap-2">
              <button
                type="button"
                onClick={() => setSidePanelTab("details")}
                className={`rounded-full px-4 py-2 text-sm font-medium ${
                  sidePanelTab === "details"
                    ? "bg-slate-900 text-white"
                    : "bg-slate-100 text-slate-700"
                }`}
              >
                Segment Details
              </button>

              <button
                type="button"
                onClick={() => setSidePanelTab("top-risk")}
                className={`rounded-full px-4 py-2 text-sm font-medium ${
                  sidePanelTab === "top-risk"
                    ? "bg-slate-900 text-white"
                    : "bg-slate-100 text-slate-700"
                }`}
              >
                Top Risk Segments
              </button>
            </div>

            {sidePanelTab === "details" ? (
              <SegmentDetailsPanel
                mode={mode}
                selectedSegment={selectedSegment}
              />
            ) : (
              <TopRiskSegmentsPanel
                clusterFilter={clusterFilter}
                onSelectSegment={setSelectedSegment}
                onOpenDetails={() => setSidePanelTab("details")}
              />
            )}
          </div>
        </div>
      </div>
    </main>
  );
}