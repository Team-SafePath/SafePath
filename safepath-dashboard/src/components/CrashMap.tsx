"use client";

import { useEffect, useMemo, useState } from "react";
import { GeoJSON, MapContainer, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";

type MapMode = "crashes" | "risk";
type RiskFilter = "all" | "top1" | "top5" | "top10";

type GeoJsonFeature = {
  type: string;
  properties: {
    segment_id: number;
    total_crashes: number;
    log_total_crashes: number;
    avg_predicted_risk: number;
    normalized_risk: number;
    segment_length: number;
    road_type: string;
  };
  geometry: GeoJSON.Geometry;
};

type GeoJsonData = {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
};

function getColor(value: number, maxValue: number) {
  const ratio = maxValue > 0 ? value / maxValue : 0;

  if (ratio > 0.85) return "#7f0000";
  if (ratio > 0.65) return "#b30000";
  if (ratio > 0.45) return "#e34a33";
  if (ratio > 0.25) return "#fc8d59";
  if (ratio > 0.10) return "#fdcc8a";
  return "#fee8c8";
}

export default function CrashMap() {
  const [data, setData] = useState<GeoJsonData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [minCrashes, setMinCrashes] = useState<number>(1);
  const [mode, setMode] = useState<MapMode>("crashes");
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");

  useEffect(() => {
    fetch("/data/segment_combined_map.geojson")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load segment map");
        return res.json();
      })
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  const maxCrashCount = useMemo(() => {
    if (!data || data.features.length === 0) return 1;
    return Math.max(...data.features.map((f) => f.properties.total_crashes));
  }, [data]);

  const maxLogCrashes = useMemo(() => {
    if (!data || data.features.length === 0) return 1;
    return Math.max(...data.features.map((f) => f.properties.log_total_crashes));
  }, [data]);

  const riskCutoffs = useMemo(() => {
    if (!data || data.features.length === 0) {
      return { top1: 0, top5: 0, top10: 0 };
    }

    const sorted = [...data.features]
      .map((f) => f.properties.avg_predicted_risk)
      .sort((a, b) => b - a);

    const cutoffAt = (fraction: number) => {
      const idx = Math.max(0, Math.ceil(sorted.length * fraction) - 1);
      return sorted[idx] ?? 0;
    };

    return {
      top1: cutoffAt(0.01),
      top5: cutoffAt(0.05),
      top10: cutoffAt(0.10),
    };
  }, [data]);

  const filteredData = useMemo(() => {
    if (!data) return null;

    if (mode === "crashes") {
      return {
        ...data,
        features: data.features.filter(
          (f) => f.properties.total_crashes >= minCrashes
        ),
      };
    }

    if (riskFilter === "all") {
      return data;
    }

    const cutoff =
      riskFilter === "top1"
        ? riskCutoffs.top1
        : riskFilter === "top5"
        ? riskCutoffs.top5
        : riskCutoffs.top10;

    return {
      ...data,
      features: data.features.filter(
        (f) => f.properties.avg_predicted_risk >= cutoff
      ),
    };
  }, [data, minCrashes, mode, riskFilter, riskCutoffs]);

  const legendTitle =
    mode === "crashes" ? "Crash Intensity" : "Predicted Risk";

  const riskFilterLabel =
    riskFilter === "all"
      ? "All segments"
      : riskFilter === "top1"
      ? "Top 1% risk"
      : riskFilter === "top5"
      ? "Top 5% risk"
      : "Top 10% risk";

  return (
    <div className="space-y-4">
      {/* Mode toggle */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => setMode("crashes")}
          className={`rounded-xl px-4 py-2 text-sm font-medium ${
            mode === "crashes"
              ? "bg-slate-900 text-white"
              : "border border-slate-300 bg-white text-slate-700"
          }`}
        >
          Historical Crash Frequency
        </button>

        <button
          onClick={() => setMode("risk")}
          className={`rounded-xl px-4 py-2 text-sm font-medium ${
            mode === "risk"
              ? "bg-slate-900 text-white"
              : "border border-slate-300 bg-white text-slate-700"
          }`}
        >
          Predicted Risk
        </button>
      </div>

      {/* Crash controls */}
      {mode === "crashes" && (
        <div className="flex flex-wrap items-center gap-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <label className="text-sm font-medium text-slate-700">
            Minimum crashes shown:{" "}
            <span className="font-semibold">{minCrashes}</span>
          </label>

          <input
            type="range"
            min={1}
            max={maxCrashCount}
            step={1}
            value={minCrashes}
            onChange={(e) => setMinCrashes(Number(e.target.value))}
            className="w-72"
          />

          <div className="text-sm text-slate-600">
            Segments shown: {filteredData?.features.length ?? 0}
          </div>
        </div>
      )}

      {/* Risk controls */}
      {mode === "risk" && (
        <div className="space-y-3">
            {/* Risk filter controls */}
            <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 p-4">
            <span className="text-sm font-medium text-slate-700">
                Highlight:
            </span>

            <button
                onClick={() => setRiskFilter("all")}
                className={`rounded-xl px-4 py-2 text-sm font-medium ${
                riskFilter === "all"
                    ? "bg-slate-900 text-white"
                    : "border border-slate-300 bg-white text-slate-700"
                }`}
            >
                Show All
            </button>

            <button
                onClick={() => setRiskFilter("top1")}
                className={`rounded-xl px-4 py-2 text-sm font-medium ${
                riskFilter === "top1"
                    ? "bg-slate-900 text-white"
                    : "border border-slate-300 bg-white text-slate-700"
                }`}
            >
                Top 1%
            </button>

            <button
                onClick={() => setRiskFilter("top5")}
                className={`rounded-xl px-4 py-2 text-sm font-medium ${
                riskFilter === "top5"
                    ? "bg-slate-900 text-white"
                    : "border border-slate-300 bg-white text-slate-700"
                }`}
            >
                Top 5%
            </button>

            <button
                onClick={() => setRiskFilter("top10")}
                className={`rounded-xl px-4 py-2 text-sm font-medium ${
                riskFilter === "top10"
                    ? "bg-slate-900 text-white"
                    : "border border-slate-300 bg-white text-slate-700"
                }`}
            >
                Top 10%
            </button>

            <div className="text-sm text-slate-600">
                {riskFilterLabel} · Segments shown: {filteredData?.features.length ?? 0}
            </div>
            </div>

            {/* Explanatory note */}
            <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600">
            <strong className="text-slate-800">How to interpret this:</strong>{" "}
            Segments are ranked by predicted crash risk from the model. Selecting
            Top 1%, 5%, or 10% highlights the highest-risk locations across the city.
            This can be used to prioritize limited safety resources toward the most
            critical areas.
            </div>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">
          {error}
        </div>
      )}

      {!filteredData && !error && (
        <div className="rounded-xl border border-slate-200 p-4 text-slate-500">
          Loading map...
        </div>
      )}

      {filteredData && (
        <div className="relative overflow-hidden rounded-2xl border border-slate-200">
          <MapContainer
            center={[40.7128, -74.006]}
            zoom={11}
            scrollWheelZoom={true}
            className="h-[700px] w-full"
          >
            <TileLayer
              attribution='&copy; OpenStreetMap contributors &copy; CARTO'
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            />

            <GeoJSON
              key={`geojson-${mode}-${riskFilter}-${minCrashes}-${filteredData.features.length}`}
              data={filteredData as GeoJSON.GeoJsonObject}
              style={(feature) => {
                const props = feature?.properties as GeoJsonFeature["properties"];

                const colorValue =
                  mode === "crashes"
                    ? props.log_total_crashes
                    : props.normalized_risk;

                const scaleMax = mode === "crashes" ? maxLogCrashes : 1;

                return {
                  color: getColor(colorValue, scaleMax),
                  weight:
                    mode === "crashes"
                      ? props.total_crashes > 50
                        ? 3
                        : 1.5
                      : riskFilter === "all"
                      ? props.avg_predicted_risk > 0.3
                        ? 3
                        : 1.5
                      : 3,
                  opacity:
                    mode === "crashes"
                      ? props.total_crashes > 50
                        ? 1
                        : 0.6
                      : riskFilter === "all"
                      ? props.avg_predicted_risk > 0.3
                        ? 1
                        : 0.7
                      : 1,
                };
              }}
              onEachFeature={(feature, layer) => {
                const props = feature.properties as GeoJsonFeature["properties"];

                layer.bindTooltip(
                  `
                  <div style="font-size: 12px;">
                    <div><strong>Segment ID:</strong> ${props.segment_id}</div>
                    <div><strong>Total Crashes:</strong> ${props.total_crashes}</div>
                    <div><strong>Predicted Risk:</strong> ${Number(
                      props.avg_predicted_risk
                    ).toFixed(4)}</div>
                    <div><strong>Road Type:</strong> ${props.road_type ?? "N/A"}</div>
                    <div><strong>Segment Length:</strong> ${Number(
                      props.segment_length
                    ).toFixed(1)}</div>
                  </div>
                  `,
                  { sticky: false }
                );
              }}
            />
          </MapContainer>

          {/* Overlay Legend */}
          <div className="pointer-events-none absolute right-4 top-4 z-[1000]">
            <div className="min-w-[220px] rounded-xl border border-slate-200 bg-white/95 px-4 py-3 shadow-md backdrop-blur">
              <div className="mb-2 text-sm font-semibold text-slate-800">
                {legendTitle}
              </div>

              <div className="flex items-center gap-2 text-xs text-slate-600">
                <span className="shrink-0">Low</span>

                <div className="flex h-3 flex-1 overflow-hidden rounded">
                  <div className="flex-1 bg-[#fee8c8]" />
                  <div className="flex-1 bg-[#fdcc8a]" />
                  <div className="flex-1 bg-[#fc8d59]" />
                  <div className="flex-1 bg-[#e34a33]" />
                  <div className="flex-1 bg-[#b30000]" />
                  <div className="flex-1 bg-[#7f0000]" />
                </div>

                <span className="shrink-0">High</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}