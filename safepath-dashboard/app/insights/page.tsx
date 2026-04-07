"use client";

import Link from "next/link";
import FeatureImportanceChart from "@/src/components/FeatureImportanceChart";
import TopRiskSegments from "@/src/components/TopRiskSegments";
import RiskTimingSummary from "@/src/components/RiskTimingSummary";

function SectionCard({
  id,
  eyebrow,
  title,
  description,
  children,
}: {
  id: string;
  eyebrow: string;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section
      id={id}
      className="scroll-mt-24 rounded-3xl border border-slate-200 bg-white p-8 shadow-sm"
    >
      <div className="mb-6 space-y-3">
        <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500">
          {eyebrow}
        </p>
        <h2 className="text-3xl font-bold tracking-tight">{title}</h2>
        <p className="max-w-3xl text-slate-600">{description}</p>
      </div>

      {children}
    </section>
  );
}

export default function InsightsPage() {
  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <div className="mb-10 grid gap-8 lg:grid-cols-[220px_minmax(0,1fr)]">
          <aside className="lg:sticky lg:top-24 lg:self-start">
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <p className="text-sm font-semibold text-slate-900">Insights</p>
              <p className="mt-1 text-sm text-slate-600">
                Explore crash risk through the Why / Where / When framework.
              </p>

              <nav className="mt-5 flex flex-col gap-3 text-sm">
                <a href="#why" className="text-slate-700 hover:text-slate-900">
                  Why
                </a>
                <a href="#where" className="text-slate-700 hover:text-slate-900">
                  Where
                </a>
                <a href="#when" className="text-slate-700 hover:text-slate-900">
                  When
                </a>
              </nav>

              <div className="mt-6 border-t border-slate-200 pt-4">
                <Link
                  href="/crash-map"
                  className="text-sm font-medium text-slate-900 underline"
                >
                  Open Crash Map
                </Link>
              </div>
            </div>
          </aside>

          <div className="space-y-8">
            <div className="rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
              <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500">
                Why · Where · When
              </p>
              <h1 className="mt-3 text-4xl font-bold tracking-tight">
                Crash Insights
              </h1>
              <p className="mt-4 max-w-4xl text-slate-600">
                SafePath approaches crash risk from three angles: why risk is
                associated with certain features, where high-risk patterns are
                concentrated across the network, and when citywide risk regimes
                shift over time.
              </p>
            </div>

            <SectionCard
              id="why"
              eyebrow="Why"
              title="Why do crashes happen?"
              description="This section explains the factors most associated with crash risk, including recent crash persistence, environmental context, and roadway characteristics."
            >
              <div className="grid gap-6 lg:grid-cols-2">
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
                  <h3 className="text-lg font-semibold">Key drivers</h3>

                  <div className="mt-4">
                    <FeatureImportanceChart />
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
                  <h3 className="text-lg font-semibold">Interpretation</h3>
                  <p className="mt-2 text-sm text-slate-600">
                    Feature importance percentages show how much each variable
                    contributes to the model’s predictions relative to the other
                    top-ranked features. Higher percentages indicate features
                    that the model relies on more heavily when distinguishing
                    between higher- and lower-risk segments.
                  </p>

                  <p className="mt-4 text-sm text-slate-600">
                    In SafePath, recent crash history and environmental context
                    emerge as the dominant drivers of risk. This suggests that
                    crash risk is shaped less by roadway structure alone and
                    more by a combination of temporal persistence and changing
                    conditions such as temperature, precipitation, and seasonal
                    timing.
                  </p>
                </div>
              </div>
            </SectionCard>

            <SectionCard
              id="where"
              eyebrow="Where"
              title="Where are crashes concentrated?"
              description="This section identifies the specific segments that the model flags as highest risk and summarizes what they have in common."
            >
              <div className="space-y-6">
                <TopRiskSegments />

                <div className="rounded-2xl border border-slate-200 bg-white p-6">
                  <h3 className="text-lg font-semibold">Interpretation</h3>

                  <div className="mt-4 grid gap-4 lg:grid-cols-2">
                    <p className="text-sm text-slate-600">
                      Rather than only showing abstract cluster averages, this
                      view surfaces the specific segments that repeatedly rank
                      highest in modeled crash risk. These are the roads most
                      relevant for prioritization when safety resources are
                      limited.
                    </p>

                    <p className="text-sm text-slate-600">
                      The top-risk group also reveals shared structural patterns
                      such as dominant road types, elevated historical crash
                      counts, and stronger overall segment risk. This makes the
                      Where question concrete: which roads stand out, and what
                      do they have in common?
                    </p>
                  </div>
                </div>
              </div>
            </SectionCard>

            <SectionCard
              id="when"
              eyebrow="When"
              title="When does risk change?"
              description="This section shows when the city is more likely to enter higher-risk conditions, using HMM regime output summarized over time."
            >
              <div className="space-y-6">
                <RiskTimingSummary />

                <div className="rounded-2xl border border-slate-200 bg-white p-6">
                  <h3 className="text-lg font-semibold">Interpretation</h3>

                  <div className="mt-4 grid gap-4 lg:grid-cols-2">
                    <p className="text-sm text-slate-600">
                      The HMM is useful because it turns daily crash conditions
                      into interpretable risk regimes rather than treating each
                      day as an isolated observation. That makes it possible to
                      identify periods where the city is operating under more
                      elevated background risk.
                    </p>

                    <p className="text-sm text-slate-600">
                      This timing view is still broad rather than event-specific:
                      it highlights monthly and regime-based changes, not exact
                      holidays or hourly traffic peaks. But it provides a much
                      clearer answer to the When question than a simple average
                      over latent states.
                    </p>
                  </div>
                </div>
              </div>
            </SectionCard>
          </div>
        </div>
      </div>
    </main>
  );
}