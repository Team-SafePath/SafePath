"use client";

import { useEffect, useMemo, useState } from "react";

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

type TopSegment = {
  segment_id: number;
  total_crashes: number;
  avg_predicted_risk: number;
  road_type: string;
  segment_length: number;
};

function cleanRoadType(roadType: string) {
  if (!roadType) return "Unknown";
  return roadType.replaceAll("_", " ");
}

export default function TopRiskSegments() {
  const [data, setData] = useState<GeoJsonData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/segment_combined_map.geojson")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load segment map data");
        return res.json();
      })
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  const topSegments = useMemo<TopSegment[]>(() => {
    if (!data) return [];

    return [...data.features]
      .sort(
        (a, b) =>
          b.properties.avg_predicted_risk - a.properties.avg_predicted_risk
      )
      .slice(0, 10)
      .map((f) => ({
        segment_id: f.properties.segment_id,
        total_crashes: f.properties.total_crashes,
        avg_predicted_risk: f.properties.avg_predicted_risk,
        road_type: cleanRoadType(f.properties.road_type),
        segment_length: f.properties.segment_length,
      }));
  }, [data]);

  const summary = useMemo(() => {
    if (!data || data.features.length === 0) return null;

    const sorted = [...data.features].sort(
      (a, b) => b.properties.avg_predicted_risk - a.properties.avg_predicted_risk
    );

    const topCount = Math.max(1, Math.ceil(sorted.length * 0.05));
    const topGroup = sorted.slice(0, topCount);

    const avg = (vals: number[]) =>
      vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;

    const roadTypeCounts = new Map<string, number>();
    topGroup.forEach((f) => {
      const key = cleanRoadType(f.properties.road_type);
      roadTypeCounts.set(key, (roadTypeCounts.get(key) ?? 0) + 1);
    });

    const dominantRoadTypes = [...roadTypeCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([road, count]) => ({
        road,
        count,
        pct: (count / topGroup.length) * 100,
      }));

    return {
      topCount,
      avgRisk: avg(topGroup.map((f) => f.properties.avg_predicted_risk)),
      avgCrashes: avg(topGroup.map((f) => f.properties.total_crashes)),
      avgLength: avg(topGroup.map((f) => f.properties.segment_length)),
      dominantRoadTypes,
    };
  }, [data]);

  if (error) {
    return <div className="text-sm text-red-600">{error}</div>;
  }

  if (!data) {
    return <div className="text-sm text-slate-500">Loading top-risk segments...</div>;
  }

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
        <h3 className="text-lg font-semibold">Top-risk segments</h3>
        <p className="mt-2 text-sm text-slate-600">
          These segments have the highest average predicted crash risk across
          the study period.
        </p>

        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full border-separate border-spacing-0 text-sm">
            <thead>
              <tr className="text-left text-slate-500">
                <th className="border-b border-slate-200 px-3 py-2 font-medium">
                  Segment ID
                </th>
                <th className="border-b border-slate-200 px-3 py-2 font-medium">
                  Predicted Risk
                </th>
                <th className="border-b border-slate-200 px-3 py-2 font-medium">
                  Total Crashes
                </th>
                <th className="border-b border-slate-200 px-3 py-2 font-medium">
                  Road Type
                </th>
                <th className="border-b border-slate-200 px-3 py-2 font-medium">
                  Length
                </th>
              </tr>
            </thead>
            <tbody>
              {topSegments.map((segment) => (
                <tr key={segment.segment_id} className="text-slate-700">
                  <td className="border-b border-slate-100 px-3 py-2">
                    {segment.segment_id}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-2 font-medium">
                    {segment.avg_predicted_risk.toFixed(4)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-2">
                    {segment.total_crashes}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-2">
                    {segment.road_type}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-2">
                    {segment.segment_length.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {summary && (
        <div className="rounded-2xl border border-slate-200 bg-white p-6">
          <h3 className="text-lg font-semibold">Shared characteristics</h3>
          <p className="mt-2 text-sm text-slate-600">
            Summary statistics for the top 5% highest-risk segments.
          </p>

          <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm text-slate-500">Top segment group</div>
              <div className="mt-2 text-2xl font-semibold">
                {summary.topCount.toLocaleString()}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                Highest-risk 5% of mapped segments
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm text-slate-500">Avg predicted risk</div>
              <div className="mt-2 text-2xl font-semibold">
                {summary.avgRisk.toFixed(4)}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                Mean model-estimated risk in the top group
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm text-slate-500">Avg crash count</div>
              <div className="mt-2 text-2xl font-semibold">
                {summary.avgCrashes.toFixed(1)}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                Historical crashes per top-risk segment
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm text-slate-500">Avg segment length</div>
              <div className="mt-2 text-2xl font-semibold">
                {summary.avgLength.toFixed(1)}
              </div>
              <div className="mt-1 text-xs text-slate-500">
                Segment length among top-risk roads
              </div>
            </div>
          </div>

          <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-4">
            <div className="text-sm font-medium text-slate-900">
              Dominant road types in the top 5%
            </div>
            <div className="mt-3 flex flex-wrap gap-3">
              {summary.dominantRoadTypes.map((item) => (
                <div
                  key={item.road}
                  className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm"
                >
                  <div className="font-medium text-slate-900">{item.road}</div>
                  <div className="mt-1 text-slate-500">
                    {item.count.toLocaleString()} segments · {item.pct.toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}