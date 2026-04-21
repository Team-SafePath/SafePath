// Portions of the React Leaflet integration and map interaction logic
// were developed with the assistance of AI tools (ChatGPT).
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

function ZoomToSelectedSegment({
  data,
  selectedSegment,
}: {
  data: SegmentFeatureCollection | null;
  selectedSegment: SelectedSegment;
}) {
  const map = useMap();

  useEffect(() => {
    if (!data || !selectedSegment) return;

    const match = data.features.find(
      (f) => f.properties.segment_id === selectedSegment.segment_id
    );

    if (!match) return;

    const layer = L.geoJSON(match as GeoJSON.GeoJsonObject);
    const bounds = layer.getBounds();

    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        padding: [60, 60],
        maxZoom: 17,
        animate: true,
      });
    }
  }, [data, selectedSegment, map]);

  return null;
}

function getContinuousColor(value: number, maxValue: number) {
  const ratio = maxValue > 0 ? value / maxValue : 0;

  if (ratio > 0.9) return "#4c1d95";
  if (ratio > 0.75) return "#6d28d9";
  if (ratio > 0.55) return "#8b5cf6";
  if (ratio > 0.35) return "#c4b5fd";
  if (ratio > 0.15) return "#e9d5ff";
  return "#f5f3ff";
}

function getHistoricalColor(value: number) {
  return getContinuousColor(value, 1);
}

function getPredictedColor(percentile: number) {
  return getContinuousColor(percentile, 1);
}

function getBinaryColor(value: number) {
  return value === 1 ? "#0f766e" : "#ccfbf1";
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

function getColorByMode(
  feature: SegmentFeature,
  mode: MapMode,
  overlay: InfrastructureOverlay,
  infrastructureMax: number,
  clusterFilter: string
) {
  const p = feature.properties;

  if (mode === "historical") {
    const crashes = Number(p.log_total_crashes ?? 0);
    return getHistoricalColor(crashes);
  }

  if (mode === "predicted") {
    const percentile = Number(p.risk_percentile ?? 0);
    return getPredictedColor(percentile);
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

function getWeightByMode(isSelected: boolean) {
  return isSelected ? 4 : 2.2;
}

function getOpacityByMode(isSelected: boolean) {
  return isSelected ? 1 : 0.9;
}

export default function CrashMapCanvas({
  mode,
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
      ? clusterFilter !== "all"
        ? "Predicted Risk Percentile · Cluster Filtered"
        : "Predicted Risk Percentile"
      : clusterFilter !== "all"
      ? "Infrastructure Cluster"
      : `${formatOverlayLabel(overlay)} Intensity`;

  const legendDescription =
    mode === "historical"
      ? "Darker lines indicate segments with more historical crashes."
      : mode === "predicted"
      ? clusterFilter !== "all"
        ? "Darker purple lines indicate segments ranked higher by the prediction model within the selected cluster."
        : "Darker purple lines indicate segments ranked higher by the prediction model."
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
        <ZoomToSelectedSegment
          data={clusterFilteredData}
          selectedSegment={selectedSegment}
        />

        <GeoJSON
          key={`geojson-${mode}-${clusterFilter}-${overlay}-${clusterFilteredData.features.length}`}
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
                clusterFilter
              ),
              weight: getWeightByMode(isSelected),
              opacity: getOpacityByMode(isSelected),
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