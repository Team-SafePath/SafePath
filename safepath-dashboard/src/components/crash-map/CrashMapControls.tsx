"use client";

import {
  InfrastructureOverlay,
  MapMode,
} from "@/src/lib/crashMap/types";

type Props = {
  mode: MapMode;
  onModeChange: (mode: MapMode) => void;
  topK: "all" | "top10" | "top5" | "top1";
  onTopKChange: (value: "all" | "top10" | "top5" | "top1") => void;
  clusterFilter: string;
  onClusterFilterChange: (value: string) => void;
  overlay: InfrastructureOverlay;
  onOverlayChange: (value: InfrastructureOverlay) => void;
};

export default function CrashMapControls({
  mode,
  onModeChange,
  topK,
  onTopKChange,
  clusterFilter,
  onClusterFilterChange,
  overlay,
  onOverlayChange,
}: Props) {
  const clusterEnabled = mode === "predicted" || mode === "infrastructure";
  const topKEnabled = mode === "predicted";
  const overlayEnabled = mode === "infrastructure";

  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div className="flex flex-col gap-2">
          <label className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
            Map Mode
          </label>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => onModeChange("historical")}
              className={`rounded-full px-4 py-2 text-sm font-medium ${
                mode === "historical"
                  ? "bg-slate-900 text-white"
                  : "bg-slate-100 text-slate-700"
              }`}
            >
              Historical Crashes
            </button>
            <button
              onClick={() => onModeChange("predicted")}
              className={`rounded-full px-4 py-2 text-sm font-medium ${
                mode === "predicted"
                  ? "bg-slate-900 text-white"
                  : "bg-slate-100 text-slate-700"
              }`}
            >
              Predicted Risk
            </button>
            <button
              onClick={() => onModeChange("infrastructure")}
              className={`rounded-full px-4 py-2 text-sm font-medium ${
                mode === "infrastructure"
                  ? "bg-slate-900 text-white"
                  : "bg-slate-100 text-slate-700"
              }`}
            >
              Infrastructure Patterns
            </button>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-3 xl:w-[760px]">
          <div>
            <label className="mb-2 block text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
              Predicted Risk Highlight
            </label>
            <select
              value={topK}
              onChange={(e) =>
                onTopKChange(e.target.value as "all" | "top10" | "top5" | "top1")
              }
              disabled={!topKEnabled}
              className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm disabled:bg-slate-100"
            >
              <option value="all">Show all segments</option>
              <option value="top10">Highlight top 10%</option>
              <option value="top5">Highlight top 5%</option>
              <option value="top1">Highlight top 1%</option>
            </select>
          </div>

          <div>
            <label className="mb-2 block text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
              Cluster Filter
            </label>
            <select
              value={clusterFilter}
              onChange={(e) => onClusterFilterChange(e.target.value)}
              disabled={!clusterEnabled}
              className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm disabled:bg-slate-100"
            >
              <option value="all">All clusters</option>
              <option value="High-Risk Persistent Segments">
                High-Risk Persistent Segments
              </option>
              <option value="Elevated Risk Corridors">
                Elevated Risk Corridors
              </option>
              <option value="Moderate-Risk Segments">
                Moderate-Risk Segments
              </option>
              <option value="Intermediate Risk Group">
                Intermediate Risk Group
              </option>
              <option value="Low-Risk Baseline">Low-Risk Baseline</option>
            </select>
          </div>

          <div>
            <label className="mb-2 block text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
              Infrastructure Overlay
            </label>
            <select
              value={overlay}
              onChange={(e) =>
                onOverlayChange(e.target.value as InfrastructureOverlay)
              }
              disabled={!overlayEnabled}
              className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm disabled:bg-slate-100"
            >
              <option value="visibility_risk_score">Visibility Risk</option>
              <option value="segment_curvature">Curvature</option>
              <option value="bearing_change_max">Bearing Change</option>
              <option value="intersection_degree_max">Intersection Degree</option>
              <option value="near_traffic_signal">Traffic Signal Proximity</option>
              <option value="near_intersection">Intersection Proximity</option>
              <option value="lanes">Lanes</option>
              <option value="maxspeed">Max Speed</option>
            </select>
          </div>
        </div>
      </div>

      {mode === "predicted" && (
        <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
          <span className="font-medium text-slate-800">How to read this:</span>{" "}
          the full network stays visible, while the selected top-risk subset is
          highlighted against the rest of the city. Cluster filters can also be
          applied here to see whether certain segment archetypes dominate the
          highest-risk predicted areas.
        </div>
      )}
    </div>
  );
}