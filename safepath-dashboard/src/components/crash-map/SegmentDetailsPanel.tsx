"use client";

import { MapMode, SelectedSegment } from "@/src/lib/crashMap/types";

type Props = {
  mode: MapMode;
  selectedSegment: SelectedSegment;
};

function formatMaybeNumber(value: number | undefined, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "N/A";
  }
  return value.toFixed(digits);
}

function formatMaybeInt(value: number | undefined) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "N/A";
  }
  return `${Math.round(value)}`;
}

function binaryLabel(value: number | undefined) {
  if (value === undefined || value === null) return "N/A";
  return value === 1 ? "Yes" : "No";
}

export default function SegmentDetailsPanel({
  mode,
  selectedSegment,
}: Props) {
  if (!selectedSegment) {
    return (
      <div className="flex h-full min-h-[640px] items-center justify-center rounded-2xl bg-slate-50 p-8 text-center">
        <div>
          <h3 className="text-lg font-semibold">Select a segment</h3>
          <p className="mt-2 text-sm text-slate-600">
            Click a street segment on the map to inspect crash history,
            predicted risk, infrastructure traits, and spatial archetype.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
          Segment Details
        </p>
        <h2 className="mt-2 text-2xl font-bold">
          Segment {selectedSegment.segment_id}
        </h2>
        <p className="mt-2 text-sm text-slate-600">
          {mode === "historical" &&
            "Viewing historical crash concentration for this segment."}
          {mode === "predicted" &&
            "Viewing model-estimated crash risk and prioritization context."}
          {mode === "infrastructure" &&
            "Viewing structural patterns and infrastructure signals for this segment."}
        </p>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-slate-600">
          Risk Summary
        </h3>
        <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-slate-500">Road type</div>
            <div className="font-medium">{selectedSegment.road_type ?? "N/A"}</div>
          </div>
          <div>
            <div className="text-slate-500">Historical crashes</div>
            <div className="font-medium">
              {formatMaybeInt(selectedSegment.total_crashes)}
            </div>
          </div>
          <div>
            <div className="text-slate-500">Predicted risk</div>
            <div className="font-medium">
              {formatMaybeNumber(selectedSegment.avg_predicted_risk, 4)}
            </div>
          </div>
          <div>
            <div className="text-slate-500">Cluster</div>
            <div className="font-medium">
              {selectedSegment.cluster_label ?? "N/A"}
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-slate-600">
          Infrastructure Profile
        </h3>
        <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-slate-500">Segment length</div>
            <div className="font-medium">
              {formatMaybeNumber(selectedSegment.segment_length, 1)}
            </div>
          </div>
          <div>
            <div className="text-slate-500">Lanes</div>
            <div className="font-medium">{formatMaybeNumber(selectedSegment.lanes, 1)}</div>
          </div>
          <div>
            <div className="text-slate-500">Max speed</div>
            <div className="font-medium">
              {formatMaybeNumber(selectedSegment.maxspeed, 1)}
            </div>
          </div>
          <div>
            <div className="text-slate-500">Visibility risk</div>
            <div className="font-medium">
              {formatMaybeNumber(selectedSegment.visibility_risk_score, 3)}
            </div>
          </div>
          <div>
            <div className="text-slate-500">Curvature</div>
            <div className="font-medium">
              {formatMaybeNumber(selectedSegment.segment_curvature, 3)}
            </div>
          </div>
          <div>
            <div className="text-slate-500">Bearing change</div>
            <div className="font-medium">
              {formatMaybeNumber(selectedSegment.bearing_change_max, 2)}
            </div>
          </div>
          <div>
            <div className="text-slate-500">Intersection degree</div>
            <div className="font-medium">
              {formatMaybeNumber(selectedSegment.intersection_degree_max, 1)}
            </div>
          </div>
          <div>
            <div className="text-slate-500">Near traffic signal</div>
            <div className="font-medium">
              {binaryLabel(selectedSegment.near_traffic_signal)}
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-slate-600">
          Why this may matter
        </h3>
        <p className="mt-3 text-sm leading-6 text-slate-600">
          This segment can be compared through three lenses: observed crash
          history, model-estimated risk, and infrastructure context. Features
          like curvature, turning complexity, speed environment, and proximity
          to signals or intersections help explain why some segments resemble
          higher-risk archetypes.
        </p>
      </div>
    </div>
  );
}