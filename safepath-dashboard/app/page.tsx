import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-white text-slate-900">
      <div className="mx-auto max-w-7xl px-6 py-16">
        <div className="max-w-4xl space-y-6">
          <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500">
            SafePath
          </p>

          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
            NYC Crash Risk Dashboard
          </h1>

          <p className="text-lg leading-8 text-slate-600">
            SafePath identifies, explains, and visualizes crash risk across New
            York City road segments using supervised prediction, spatial
            clustering, and temporal regime analysis.
          </p>
        </div>

        <div className="mt-12 grid gap-6 md:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 p-6 shadow-sm">
            <h2 className="text-lg font-semibold">Crash Map</h2>
            <p className="mt-2 text-sm text-slate-600">
              Explore historical crash frequency and modeled segment risk across
              the city.
            </p>
            <Link
              href="/crash-map"
              className="mt-4 inline-block text-sm font-medium text-slate-900 underline"
            >
              Open map
            </Link>
          </div>

          <div className="rounded-2xl border border-slate-200 p-6 shadow-sm">
            <h2 className="text-lg font-semibold">Insights</h2>
            <p className="mt-2 text-sm text-slate-600">
              Understand why crashes happen, where they cluster, and when risk
              shifts over time.
            </p>
            <Link
              href="/insights"
              className="mt-4 inline-block text-sm font-medium text-slate-900 underline"
            >
              Open insights
            </Link>
          </div>

          <div className="rounded-2xl border border-slate-200 p-6 shadow-sm">
            <h2 className="text-lg font-semibold">Project Scope</h2>
            <p className="mt-2 text-sm text-slate-600">
              Street-level crash risk analysis combining geospatial,
              environmental, and temporal context.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}