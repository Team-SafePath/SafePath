"use client";

import { useEffect, useMemo, useState } from "react";
import { SelectedSegment } from "@/src/lib/crashMap/types";

type GeoJsonFeature = {
  type: string;
  properties: {
    segment_id?: number;
    total_crashes?: number;
    avg_predicted_risk?: number;
    risk_percentile?: number;
    cluster_label?: string;
    road_type?: string;
    maxspeed?: number;
    lanes?: number;
    visibility_risk_score?: number;
    near_intersection?: number;
    near_traffic_signal?: number;
    segment_length?: number;
  };
  geometry: GeoJSON.Geometry;
};

type GeoJsonCollection = {
  type: "FeatureCollection";
  features: GeoJsonFeature[];
};

type RankedSegment = SelectedSegment & {
  total_crashes?: number;
  avg_predicted_risk?: number;
  risk_percentile?: number;
  cluster_label?: string;
  road_type?: string;
  maxspeed?: number;
  lanes?: number;
  visibility_risk_score?: number;
  near_intersection?: number;
  near_traffic_signal?: number;
  segment_length?: number;
};

type Props = {
  clusterFilter: string;
  onSelectSegment: (segment: SelectedSegment) => void;
  onOpenDetails: () => void;
};

const MAP_DATA_URL = process.env.NEXT_PUBLIC_SEGMENT_MAP_URL;

function fmt(value: number | undefined, digits = 2) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}

function pct(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

export default function TopRiskSegmentsPanel({
  clusterFilter,
  onSelectSegment,
  onOpenDetails,
}: Props) {
  const [data, setData] = useState<GeoJsonCollection | null>(null);
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
          setData(json as GeoJsonCollection);
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

  const rows = useMemo<RankedSegment[]>(() => {
    if (!data) return [];

    let features = data.features.filter(
      (f) =>
        f.properties.segment_id !== undefined &&
        f.properties.risk_percentile !== undefined
    );

    if (clusterFilter !== "all") {
      features = features.filter(
        (f) => (f.properties.cluster_label ?? "") === clusterFilter
      );
    }

    return [...features]
      .sort(
        (a, b) =>
          Number(b.properties.risk_percentile ?? 0) -
          Number(a.properties.risk_percentile ?? 0)
      )
      .slice(0, 25)
      .map((f) => ({
        segment_id: f.properties.segment_id as number,
        total_crashes: f.properties.total_crashes,
        avg_predicted_risk: f.properties.avg_predicted_risk,
        risk_percentile: f.properties.risk_percentile,
        cluster_label: f.properties.cluster_label,
        road_type: f.properties.road_type,
        maxspeed: f.properties.maxspeed,
        lanes: f.properties.lanes,
        visibility_risk_score: f.properties.visibility_risk_score,
        near_intersection: f.properties.near_intersection,
        near_traffic_signal: f.properties.near_traffic_signal,
        segment_length: f.properties.segment_length,
      }));
  }, [data, clusterFilter]);

  const displayError = configError || error;

  if (displayError) {
    return <div className="text-sm text-red-600">{displayError}</div>;
  }

  if (!rows.length) {
    return (
      <div className="space-y-3">
        <h3 className="text-lg font-semibold">Top Risk Segments</h3>
        <p className="text-sm text-slate-500">
          Loading ranked segment list...
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-[760px] flex-col">
      <div className="shrink-0">
        <h3 className="text-lg font-semibold">Top Risk Segments</h3>
        <p className="mt-1 text-sm text-slate-600">
          Ranked by predicted risk percentile
          {clusterFilter !== "all" ? ` within ${clusterFilter}` : ""}.
        </p>
      </div>

      <div className="mt-4 flex-1 space-y-3 overflow-y-auto pr-1">
        {rows.map((p, idx) => (
          <button
            key={p.segment_id}
            type="button"
            onClick={() => {
              onSelectSegment(p);
              onOpenDetails();
            }}
            className="w-full rounded-2xl border border-slate-200 bg-slate-50 p-4 text-left transition hover:border-slate-300 hover:bg-white"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-xs font-medium uppercase tracking-[0.14em] text-slate-500">
                  Rank #{idx + 1}
                </div>
                <div className="mt-1 text-sm font-semibold text-slate-900">
                  Segment {p.segment_id}
                </div>
                <div className="mt-1 text-sm text-slate-600">
                  {p.cluster_label ?? "Unknown cluster"} · {p.road_type ?? "N/A"}
                </div>
              </div>

              <div className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700">
                {pct(p.risk_percentile)}
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div>
                <div className="text-slate-500">Predicted risk</div>
                <div className="font-medium">{fmt(p.avg_predicted_risk, 4)}</div>
              </div>

              <div>
                <div className="text-slate-500">Total crashes</div>
                <div className="font-medium">{fmt(p.total_crashes, 0)}</div>
              </div>

              <div>
                <div className="text-slate-500">Max speed</div>
                <div className="font-medium">{fmt(p.maxspeed, 1)}</div>
              </div>

              <div>
                <div className="text-slate-500">Lanes</div>
                <div className="font-medium">{fmt(p.lanes, 1)}</div>
              </div>

              <div>
                <div className="text-slate-500">Visibility risk</div>
                <div className="font-medium">
                  {fmt(p.visibility_risk_score, 3)}
                </div>
              </div>

              <div>
                <div className="text-slate-500">Near signal</div>
                <div className="font-medium">
                  {Number(p.near_traffic_signal ?? 0) === 1 ? "Yes" : "No"}
                </div>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}