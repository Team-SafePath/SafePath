"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { GeoJSON, MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import {
  InfrastructureOverlay,
  MapMode,
  SelectedSegment,
  SegmentFeature,
  SegmentFeatureCollection,
} from "@/src/lib/crashMap/types";

type Props = {
  mode: MapMode;
  topK: "all" | "top10" | "top5" | "top1";
  clusterFilter: string;
  overlay: InfrastructureOverlay;
  selectedSegment: SelectedSegment;
  onSelectSegment: (segment: SelectedSegment) => void;
};

const MAP_DATA_URL = process.env.NEXT_PUBLIC_SEGMENT_MAP_URL;

function FitBoundsOnce({ data }: { data: SegmentFeatureCollection | null }) {
  const map = useMap();
  const hasFitRef = useRef(false);

  useEffect(() => {
    if (!data || data.features.length === 0 || hasFitRef.current) return;

    const timer = window.setTimeout(() => {
      const layer = L.geoJSON(data as GeoJSON.GeoJsonObject);
      const bounds = layer.getBounds();

      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [20, 20], animate: false });
        hasFitRef.current = true;
      }
    }, 0);

    return () => window.clearTimeout(timer);
  }, [data, map]);

  return null;
}

function getContinuousColor(value: number, maxValue: number) {
  const ratio = maxValue > 0 ? value / maxValue : 0;

  if (ratio > 0.85) return "#7f0000";
  if (ratio > 0.65) return "#b30000";
  if (ratio > 0.45) return "#e34a33";
  if (ratio > 0.25) return "#fc8d59";
  if (ratio > 0.1) return "#fdcc8a";
  return "#fee8c8";
}

function getBinaryColor(value: number) {
  return value === 1 ? "#0f766e" : "#ccfbf1";
}

function getMutedColor() {
  return "#d1d5db";
}

function getClusterColor(label: string | undefined) {
  switch (label) {
    case "High-Risk Persistent Segments":
      return "#7f1d1d";
    case "Elevated Risk Corridors":
      return "#b45309";
    case "Moderate-Risk Segments":
      return "#2563eb";
    case "Intermediate Risk Group":
      return "#7c3aed";
    case "Low-Risk Baseline":
      return "#475569";
    default:
      return "#94a3b8";
  }
}

function getInfrastructureValue(
  feature: SegmentFeature,
  overlay: InfrastructureOverlay
) {
  const p = feature.properties;
  const raw = p[overlay];

  if (raw === undefined || raw === null || Number.isNaN(Number(raw))) {
    return 0;
  }

  return Number(raw);
}

function getInfrastructureMax(
  data: SegmentFeatureCollection | null,
  overlay: InfrastructureOverlay
) {
  if (!data || data.features.length === 0) return 1;

  let max = 0;
  for (const feature of data.features) {
    const value = getInfrastructureValue(feature, overlay);
    if (value > max) max = value;
  }

  return max > 0 ? max : 1;
}

function formatOverlayLabel(overlay: InfrastructureOverlay) {
  switch (overlay) {
    case "visibility_risk_score":
      return "Visibility Risk";
    case "segment_curvature":
      return "Curvature";
    case "bearing_change_max":
      return "Bearing Change";
    case "intersection_degree_max":
      return "Intersection Degree";
    case "near_traffic_signal":
      return "Near Traffic Signal";
    case "near_intersection":
      return "Near Intersection";
    case "lanes":
      return "Lanes";
    case "maxspeed":
      return "Max Speed";
    default:
      return overlay;
  }
}

function formatOverlayValue(
  overlay: InfrastructureOverlay,
  value: number | undefined
) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "N/A";
  }

  const numericValue = Number(value);

  if (overlay === "near_traffic_signal" || overlay === "near_intersection") {
    return numericValue === 1 ? "Yes" : "No";
  }

  if (
    overlay === "lanes" ||
    overlay === "maxspeed" ||
    overlay === "intersection_degree_max"
  ) {
    return numericValue.toFixed(1);
  }

  return numericValue.toFixed(3);
}

function passesClusterFilter(
  feature: SegmentFeature,
  mode: MapMode,
  clusterFilter: string
) {
  if ((mode !== "predicted" && mode !== "infrastructure") || clusterFilter === "all") {
    return true;
  }

  return (feature.properties.cluster_label ?? "") === clusterFilter;
}

function isHighlightedTopRisk(
  feature: SegmentFeature,
  mode: MapMode,
  topK: "all" | "top10" | "top5" | "top1"
) {
  if (mode !== "predicted" || topK === "all") return true;

  const p = feature.properties;

  if (topK === "top10") return Number(p.is_top10_risk ?? 0) === 1;
  if (topK === "top5") return Number(p.is_top5_risk ?? 0) === 1;
  if (topK === "top1") return Number(p.is_top1_risk ?? 0) === 1;

  return true;
}

function getColorByMode(
  feature: SegmentFeature,
  mode: MapMode,
  overlay: InfrastructureOverlay,
  infrastructureMax: number,
  clusterFilter: string,
  topK: "all" | "top10" | "top5" | "top1"
) {
  const p = feature.properties;

  if (mode === "historical") {
    const crashes = Number(p.log_total_crashes ?? 0);
    return getContinuousColor(crashes, 1);
  }

  if (mode === "predicted") {
    const highlighted = isHighlightedTopRisk(feature, mode, topK);
    if (!highlighted) return getMutedColor();

    const pct = Number(p.risk_percentile ?? 0);
    return getContinuousColor(pct, 1);
  }

  if (clusterFilter !== "all") {
    return getClusterColor(p.cluster_label);
  }

  const value = getInfrastructureValue(feature, overlay);

  if (overlay === "near_traffic_signal" || overlay === "near_intersection") {
    return getBinaryColor(value);
  }

  return getContinuousColor(value, infrastructureMax);
}

function getWeightByMode(
  feature: SegmentFeature,
  mode: MapMode,
  topK: "all" | "top10" | "top5" | "top1",
  isSelected: boolean
) {
  if (isSelected) return 4;

  if (mode === "predicted" && topK !== "all") {
    return isHighlightedTopRisk(feature, mode, topK) ? 2.8 : 1.2;
  }

  return 2.2;
}

function getOpacityByMode(
  feature: SegmentFeature,
  mode: MapMode,
  topK: "all" | "top10" | "top5" | "top1",
  isSelected: boolean
) {
  if (isSelected) return 1;

  if (mode === "predicted" && topK !== "all") {
    return isHighlightedTopRisk(feature, mode, topK) ? 1 : 0.22;
  }

  return 0.9;
}

export default function CrashMapCanvas({
  mode,
  topK,
  clusterFilter,
  overlay,
  selectedSegment,
  onSelectSegment,
}: Props) {
  const [data, setData] = useState<SegmentFeatureCollection | null>(null);
  const [error, setError] = useState<string | null>(null);

  const configError = !MAP_DATA_URL
    ? "NEXT_PUBLIC_SEGMENT_MAP_URL is not defined. Set it in Vercel and .env.local."
    : null;

  useEffect(() => {
    if (!MAP_DATA_URL) return;

    let cancelled = false;

    fetch(MAP_DATA_URL)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to load segment map data from ${MAP_DATA_URL}`);
        }
        return res.json();
      })
      .then((json) => {
        if (!cancelled) {
          setData(json as SegmentFeatureCollection);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err.message);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const clusterFilteredData = useMemo(() => {
    if (!data) return null;

    return {
      ...data,
      features: data.features.filter((feature) =>
        passesClusterFilter(feature, mode, clusterFilter)
      ),
    };
  }, [data, mode, clusterFilter]);

  const infrastructureMax = useMemo(() => {
    return getInfrastructureMax(clusterFilteredData, overlay);
  }, [clusterFilteredData, overlay]);

  const legendTitle =
    mode === "historical"
      ? "Historical Crash Intensity"
      : mode === "predicted"
      ? topK === "all"
        ? "Predicted Risk Percentile"
        : `Predicted Risk · ${
            topK === "top10" ? "Top 10%" : topK === "top5" ? "Top 5%" : "Top 1%"
          } Highlight`
      : clusterFilter !== "all"
      ? "Infrastructure Cluster"
      : `${formatOverlayLabel(overlay)} Intensity`;

  const legendDescription =
    mode === "historical"
      ? "Darker lines indicate segments with more historical crashes."
      : mode === "predicted"
      ? topK === "all"
        ? "Darker lines indicate segments ranked higher by the prediction model."
        : "Highlighted segments belong to the selected top-risk slice, while the rest of the network remains visible for context."
      : clusterFilter !== "all"
      ? "Color reflects the selected spatial archetype."
      : overlay === "near_traffic_signal" || overlay === "near_intersection"
      ? "Teal highlights indicate where this binary infrastructure condition is present."
      : `Darker lines indicate higher values for ${formatOverlayLabel(
          overlay
        ).toLowerCase()}.`;

  const displayError = configError || error;

  if (displayError) {
    return <div className="p-6 text-sm text-red-600">{displayError}</div>;
  }

  if (!clusterFilteredData) {
    return <div className="p-6 text-sm text-slate-500">Loading map...</div>;
  }

  return (
    <div className="relative h-[760px] w-full overflow-hidden rounded-2xl bg-slate-100">
      <MapContainer
        center={[40.7128, -74.006]}
        zoom={11}
        scrollWheelZoom
        preferCanvas
        zoomAnimation={false}
        fadeAnimation={false}
        markerZoomAnimation={false}
        className="h-full w-full"
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors &copy; CARTO"
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          subdomains={["a", "b", "c", "d"]}
          maxZoom={19}
          keepBuffer={4}
          updateWhenIdle
        />

        <FitBoundsOnce data={data} />

        <GeoJSON
          data={clusterFilteredData as GeoJSON.GeoJsonObject}
          style={(feature) => {
            const f = feature as unknown as SegmentFeature;
            const isSelected =
              selectedSegment?.segment_id === f.properties.segment_id;

            return {
              color: getColorByMode(
                f,
                mode,
                overlay,
                infrastructureMax,
                clusterFilter,
                topK
              ),
              weight: getWeightByMode(f, mode, topK, isSelected),
              opacity: getOpacityByMode(f, mode, topK, isSelected),
            };
          }}
          onEachFeature={(feature, layer) => {
            const f = feature as unknown as SegmentFeature;
            const p = f.properties;

            layer.on({
              click: () => onSelectSegment(p),
            });

            layer.bindTooltip(
              `
                <div style="font-size:12px; line-height:1.4;">
                  <div><strong>Segment:</strong> ${p.segment_id}</div>
                  <div><strong>Road type:</strong> ${p.road_type ?? "N/A"}</div>
                  <div><strong>Crashes:</strong> ${p.total_crashes ?? "N/A"}</div>
                  <div><strong>Predicted Risk:</strong> ${
                    p.avg_predicted_risk !== undefined
                      ? Number(p.avg_predicted_risk).toFixed(4)
                      : "N/A"
                  }</div>
                  <div><strong>Risk Percentile:</strong> ${
                    p.risk_percentile !== undefined
                      ? `${(Number(p.risk_percentile) * 100).toFixed(1)}%`
                      : "N/A"
                  }</div>
                  <div><strong>${formatOverlayLabel(overlay)}:</strong> ${formatOverlayValue(
                    overlay,
                    p[overlay]
                  )}</div>
                  <div><strong>Cluster:</strong> ${p.cluster_label ?? "N/A"}</div>
                </div>
              `,
              { sticky: true }
            );
          }}
        />
      </MapContainer>

      <div className="pointer-events-none absolute bottom-4 left-4 z-[1000] rounded-xl border border-slate-200 bg-white/95 px-4 py-3 shadow-md">
        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
          {legendTitle}
        </div>
        <div className="mt-2 text-xs text-slate-700">{legendDescription}</div>
      </div>
    </div>
  );
}