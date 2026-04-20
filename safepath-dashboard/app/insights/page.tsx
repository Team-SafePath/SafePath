import Link from "next/link";
import FeatureImportanceChart from "@/src/components/insights/FeatureImportanceChart";
import GMMClusterArchetypes from "@/src/components/insights/GMMClusterArchetypes";
import HMMRegimeSummary from "@/src/components/insights/HMMRegimeSummary";
import ModelTakeaways from "@/src/components/insights/ModelTakeaways";
import LimitationsPanel from "@/src/components/insights/LimitationsPanel";
import ClusterFeatureHeatmap from "@/src/components/insights/ClusterFeatureHeatmap";
import KeyTakeawaysPanel from "@/src/components/insights/KeyTakeawaysPanel";

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
                Final model findings organized around Why, Where, and When.
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
                <a href="#so-what" className="text-slate-700 hover:text-slate-900">
                  So What
                </a>
                <a
                  href="#limitations"
                  className="text-slate-700 hover:text-slate-900"
                >
                  Limitations
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
                SafePath Modeling Summary
              </p>
              <h1 className="mt-3 text-4xl font-bold tracking-tight">
                Why · Where · When
              </h1>
              <p className="mt-4 max-w-4xl text-slate-600">
                SafePath combines supervised and unsupervised modeling to explain
                crash risk from three angles: why certain features drive risk,
                where high-risk segment archetypes are concentrated, and when the
                city shifts into elevated crash regimes.
              </p>
            </div>

            <KeyTakeawaysPanel />
            
            <SectionCard
              id="why"
              eyebrow="Why"
              title="Why does crash risk rise?"
              description="The supervised model highlights which structural, environmental, and recent-history features matter most when ranking segment-level crash risk."
            >
              <div className="space-y-6">
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
                  <h3 className="text-lg font-semibold">
                    Top predictive drivers
                  </h3>
                  <p className="mt-2 text-sm text-slate-600">
                    Feature importance values show which variables the
                    infrastructure-aware LightGBM model relied on most when
                    separating higher-risk from lower-risk segments.
                  </p>

                  <div className="mt-4">
                    <FeatureImportanceChart />
                  </div>
                </div>

                <div className="grid gap-6 lg:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200 bg-white p-6">
                    <h3 className="text-lg font-semibold">Interpretation</h3>
                    <p className="mt-3 text-sm leading-6 text-slate-600">
                      Risk is not explained by crash history alone. Structural
                      variables such as segment curvature, turning complexity,
                      lane count, speed environment, and a composite visibility
                      risk score all emerged as important predictors alongside
                      weather.
                    </p>
                    <p className="mt-3 text-sm leading-6 text-slate-600">
                      That means the model is now capturing physical road
                      conditions that help explain why certain segments remain
                      riskier even before a specific crash occurs.
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-white p-6">
                    <h3 className="text-lg font-semibold">
                      Key structural takeaway
                    </h3>
                    <p className="mt-3 text-sm leading-6 text-slate-600">
                      The strongest infrastructure signals point to segments that
                      are more difficult to navigate or interpret: curved
                      segments, sharper bearing changes, higher visibility risk,
                      and more complex intersection environments. These features
                      make crash risk more explainable in engineering terms, not
                      just statistical terms.
                    </p>
                  </div>
                </div>
              </div>
            </SectionCard>

            <SectionCard
              id="where"
              eyebrow="Where"
              title="Where are high-risk patterns concentrated?"
              description="The spatial clustering model groups segments into recurring archetypes that capture different kinds of road environments, including persistent residential risk and elevated corridor risk."
            >
              <GMMClusterArchetypes />
              <ClusterFeatureHeatmap />
            </SectionCard>

            <SectionCard
              id="when"
              eyebrow="When"
              title="When does the city enter elevated crash regimes?"
              description="The HMM summarizes daily citywide conditions into recurring temporal regimes, distinguishing normal operating periods from more persistent high-risk periods."
            >
              <HMMRegimeSummary />
            </SectionCard>

            <SectionCard
              id="so-what"
              eyebrow="So What"
              title="What should stakeholders take away?"
              description="These models are most useful as a prioritization and interpretation tool: they help identify which segments deserve attention, what kinds of segment environments are repeatedly risky, and when citywide conditions become more elevated."
            >
              <ModelTakeaways />
            </SectionCard>

            <SectionCard
              id="limitations"
              eyebrow="Limitations"
              title="What this system does not claim"
              description="SafePath is a predictive and exploratory system, not a causal model. These outputs should support prioritization and diagnosis, not be treated as proof of cause."
            >
              <LimitationsPanel />
            </SectionCard>
          </div>
        </div>
      </div>
    </main>
  );
}