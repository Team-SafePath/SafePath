export default function LimitationsPanel() {
  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
          <h3 className="text-lg font-semibold">Predictive, not causal</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            These models identify patterns associated with crash risk. They do
            not prove that any single feature causes crashes, and they should
            not be interpreted as a direct estimate of intervention effect.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
          <h3 className="text-lg font-semibold">Crash data has reporting limits</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            Reported crash records may understate incidents in some contexts and
            may reflect differences in reporting or data quality across places
            and time periods.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
          <h3 className="text-lg font-semibold">Temporal granularity is daily</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            The current temporal regime model operates at the day level. It
            captures citywide periods of elevated risk, but it does not yet
            isolate hour-of-day or rush-hour effects.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
          <h3 className="text-lg font-semibold">Use for prioritization</h3>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            SafePath is best used to support prioritization, diagnosis, and
            investigation. It should complement engineering review and domain
            expertise rather than replace them.
          </p>
        </div>
      </div>
    </div>
  );
}