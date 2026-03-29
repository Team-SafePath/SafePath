export default function Home() {
  return (
    <main className="min-h-screen bg-white text-slate-900">
      <div className="mx-auto max-w-6xl px-6 py-16">
        <div className="space-y-6">
          <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500">
            SafePath
          </p>

          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
            NYC Crash Risk Analysis Dashboard
          </h1>

          <p className="max-w-3xl text-lg leading-8 text-slate-600">
            SafePath combines supervised and unsupervised modeling to identify,
            explain, and visualize urban crash risk across New York City road
            segments.
          </p>
        </div>

        <div className="mt-12 grid gap-6 md:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 p-6 shadow-sm">
            <h2 className="text-lg font-semibold">Supervised Modeling</h2>
            <p className="mt-2 text-sm text-slate-600">
              LightGBM full-panel crash risk modeling using infrastructure,
              weather, and temporal features.
            </p>
          </div>

          <div className="rounded-2xl border border-slate-200 p-6 shadow-sm">
            <h2 className="text-lg font-semibold">Spatial Archetypes</h2>
            <p className="mt-2 text-sm text-slate-600">
              Segment clustering to identify groups of roads with similar crash
              behavior and structural patterns.
            </p>
          </div>

          <div className="rounded-2xl border border-slate-200 p-6 shadow-sm">
            <h2 className="text-lg font-semibold">Temporal Risk Regimes</h2>
            <p className="mt-2 text-sm text-slate-600">
              HMM-based temporal states that capture citywide shifts between low,
              moderate, and high-risk periods.
            </p>
          </div>
        </div>

        <div className="mt-10">
          <a
            href="/overview"
            className="inline-flex rounded-xl bg-slate-900 px-5 py-3 text-sm font-medium text-white hover:bg-slate-700"
          >
            Open Dashboard Overview
          </a>
        </div>
      </div>
    </main>
  );
}