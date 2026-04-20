import Link from "next/link";

function SectionCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
      <h2 className="text-2xl font-bold tracking-tight text-slate-900">
        {title}
      </h2>
      <div className="mt-4 space-y-4 text-slate-600">{children}</div>
    </section>
  );
}

function NavCard({
  href,
  eyebrow,
  title,
  description,
  cta,
}: {
  href: string;
  eyebrow: string;
  title: string;
  description: string;
  cta: string;
}) {
  return (
    <Link
      href={href}
      className="group rounded-3xl border border-slate-200 bg-white p-8 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
    >
      <p className="text-sm font-medium uppercase tracking-[0.18em] text-slate-500">
        {eyebrow}
      </p>
      <h3 className="mt-3 text-2xl font-bold tracking-tight text-slate-900">
        {title}
      </h3>
      <p className="mt-4 text-slate-600">{description}</p>
      <div className="mt-6 text-sm font-semibold text-slate-900 underline underline-offset-4">
        {cta}
      </div>
    </Link>
  );
}

export default function HomePage() {
  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <section className="rounded-3xl border border-slate-200 bg-white p-10 shadow-sm">
          <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500">
            SafePath
          </p>
          <h1 className="mt-3 max-w-4xl text-4xl font-bold tracking-tight sm:text-5xl">
            Understanding crash risk through street design, spatial patterns,
            and temporal change
          </h1>
          <p className="mt-5 max-w-4xl text-lg leading-8 text-slate-600">
            SafePath is a street-segment crash risk analysis project focused on
            explaining not just where crashes have happened, but also why some
            parts of the network appear riskier and when citywide conditions
            shift into elevated-risk periods.
          </p>
        </section>

        <div className="mt-8 grid gap-8">
          <SectionCard title="Project Scope">
            <p>
              This project combines crash records, weather information, and
              road-network infrastructure features to study segment-level crash
              risk across New York City. The goal is not only to identify
              historically dangerous locations, but also to better understand
              the structural and temporal conditions associated with elevated
              risk.
            </p>

            <p>
              The modeling workflow brings together three perspectives. A
              supervised model helps explain <strong>why</strong> risk rises by
              identifying important predictive features such as curvature,
              visibility risk, speed environment, and weather. A clustering
              model helps explain <strong>where</strong> different kinds of
              risky road environments appear, such as persistent residential
              risk versus elevated corridor risk. A temporal regime model helps
              explain <strong>when</strong> the city shifts into broader
              high-risk periods.
            </p>

            <p>
              The dashboard is organized around those same ideas. The{" "}
              <Link
                href="/crash-map"
                className="font-medium text-slate-900 underline underline-offset-4"
              >
                Crash Map
              </Link>{" "}
              supports interactive exploration of historical crashes, predicted
              risk, and infrastructure-based patterns. The{" "}
              <Link
                href="/insights"
                className="font-medium text-slate-900 underline underline-offset-4"
              >
                Insights
              </Link>{" "}
              page summarizes the major findings from the final modeling stack
              using the Why, Where, and When framework.
            </p>
          </SectionCard>

          <div className="grid gap-6 md:grid-cols-2">
            <NavCard
              href="/crash-map"
              eyebrow="Explore"
              title="Crash Map"
              description="Interact with the street network to compare historical crashes, predicted risk, and infrastructure-related patterns."
              cta="Open Crash Map"
            />
            <NavCard
              href="/insights"
              eyebrow="Understand"
              title="Insights"
              description="Review the main findings from the supervised model, spatial clustering, and temporal regime analysis."
              cta="View Insights"
            />
          </div>
        </div>
      </div>
    </main>
  );
}