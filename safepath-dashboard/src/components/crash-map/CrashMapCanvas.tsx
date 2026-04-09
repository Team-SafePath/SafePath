"use client";

import { useEffect, useMemo, useState } from "react";
import {
  GeoJSON,
  MapContainer,
  TileLayer,
  useMap,
} from "react-leaflet";
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

function FitBounds({ data }: { data: SegmentFeatureCollection | null }) {
  const map = useMap();

  useEffect(() => {
    if (!data || data.features.length === 0) return;

    const layer = L.geoJSON(data as GeoJSON.GeoJsonObject);
    const bounds = layer.getBounds();

    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [20, 20] });
    }
  }, [data, map]);

  return null;
}

function ForceResizeFix() {
  const map = useMap();

  useEffect(() => {
    const t = window.setTimeout(() => {
      map.invalidateSize();
    }, 200);

    return () => window.clearTimeout(t);
  }, [map]);

  return null;
}

function getColorByMode(
  feature: SegmentFeature,
  mode: MapMode,
  overlay: InfrastructureOverlay
) {
  const p = feature.properties;

  if (mode === "historical") {
    const crashes = p.total_crashes ?? 0;
    if (crashes > 150) return "#7f1d1d";
    if (crashes > 100) return "#b91c1c";
    if (crashes > 60) return "#ef4444";
    if (crashes > 30) return "#fca5a5";
    return "#fee2e2";
  }

  if (mode === "predicted") {
    const risk = p.avg_predicted_risk ?? 0;
    if (risk > 0.20) return "#581c87";
    if (risk > 0.15) return "#7e22ce";
    if (risk > 0.10) return "#a855f7";
    if (risk > 0.05) return "#d8b4fe";
    return "#f3e8ff";
  }

  const value = p[overlay] ?? 0;

  if (overlay === "near_traffic_signal") {
    return value === 1 ? "#0f766e" : "#ccfbf1";
  }

  if (overlay === "lanes") {
    if (value >= 4) return "#1d4ed8";
    if (value >= 3) return "#60a5fa";
    if (value >= 2) return "#bfdbfe";
    return "#eff6ff";
  }

  if (overlay === "maxspeed") {
    if (value >= 45) return "#92400e";
    if (value >= 35) return "#d97706";
    if (value >= 25) return "#fbbf24";
    return "#fef3c7";
  }

  if (value >= 0.6) return "#14532d";
  if (value >= 0.3) return "#16a34a";
  if (value >= 0.15) return "#86efac";
  return "#dcfce7";
}

function passesFilters(
  feature: SegmentFeature,
  mode: MapMode,
  topK: "all" | "top10" | "top5" | "top1",
  clusterFilter: string
) {
  const p = feature.properties;

  if (mode === "predicted") {
    const risk = p.avg_predicted_risk ?? 0;
    if (topK === "top10" && risk < 0.10) return false;
    if (topK === "top5" && risk < 0.15) return false;
    if (topK === "top1" && risk < 0.20) return false;
  }

  if (mode === "infrastructure" && clusterFilter !== "all") {
    if ((p.cluster_label ?? "") !== clusterFilter) return false;
  }

  return true;
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

  useEffect(() => {
    fetch("/data/segment_combined_map.geojson")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load segment map data");
        return res.json();
      })
      .then((json) => setData(json))
      .catch((err) => setError(err.message));
  }, []);

  const filteredData = useMemo(() => {
    if (!data) return null;

    return {
      ...data,
      features: data.features.filter((feature) =>
        passesFilters(feature, mode, topK, clusterFilter)
      ),
    };
  }, [data, mode, topK, clusterFilter]);

  if (error) {
    return <div className="p-6 text-sm text-red-600">{error}</div>;
  }

  if (!filteredData) {
    return <div className="p-6 text-sm text-slate-500">Loading map...</div>;
  }

  return (
    <div className="relative h-[760px] w-full overflow-hidden rounded-2xl bg-slate-100">
      <MapContainer
        center={[40.7128, -74.006]}
        zoom={11}
        scrollWheelZoom
        preferCanvas={true}
        className="h-full w-full"
      >
        <TileLayer
          attribution='&copy; OpenStreetMap contributors &copy; CARTO'
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          subdomains={["a", "b", "c", "d"]}
          maxZoom={19}
          keepBuffer={4}
          updateWhenIdle={true}
        />

        <ForceResizeFix />
        <FitBounds data={filteredData} />

        <GeoJSON
          data={filteredData as GeoJSON.GeoJsonObject}
          style={(feature) => {
            const f = feature as unknown as SegmentFeature;
            const isSelected =
              selectedSegment?.segment_id === f.properties.segment_id;

            return {
              color: getColorByMode(f, mode, overlay),
              weight: isSelected ? 4 : 2.2,
              opacity: 1,
            };
          }}
          onEachFeature={(feature, layer) => {
            const f = feature as unknown as SegmentFeature;

            layer.on({
              click: () => onSelectSegment(f.properties),
            });

            const p = f.properties;
            layer.bindTooltip(
              `
                <div style="font-size:12px;">
                  <div><strong>Segment:</strong> ${p.segment_id}</div>
                  <div><strong>Road type:</strong> ${p.road_type ?? "N/A"}</div>
                  <div><strong>Crashes:</strong> ${p.total_crashes ?? "N/A"}</div>
                  <div><strong>Risk:</strong> ${
                    p.avg_predicted_risk !== undefined
                      ? Number(p.avg_predicted_risk).toFixed(4)
                      : "N/A"
                  }</div>
                </div>
              `,
              { sticky: true }
            );
          }}
        />
      </MapContainer>

      <div className="pointer-events-none absolute bottom-4 left-4 z-[1000] rounded-xl border border-slate-200 bg-white/95 px-4 py-3 shadow-md">
        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
          Legend
        </div>
        <div className="mt-2 text-xs text-slate-700">
          {mode === "historical" && "Darker red = more historical crashes"}
          {mode === "predicted" && "Darker purple = higher predicted risk"}
          {mode === "infrastructure" &&
            "Color reflects the selected infrastructure overlay"}
        </div>
      </div>
    </div>
  );
}