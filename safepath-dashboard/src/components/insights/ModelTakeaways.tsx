export default function ModelTakeaways() {
  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
        <h3 className="text-lg font-semibold">Prioritize by ranking</h3>
        <p className="mt-3 text-sm leading-6 text-slate-600">
          The supervised model is most useful as a ranking tool. Instead of
          treating every segment the same, stakeholders can focus attention on
          the highest-risk slice of the network first.
        </p>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
        <h3 className="text-lg font-semibold">Different roads fail differently</h3>
        <p className="mt-3 text-sm leading-6 text-slate-600">
          The clustering analysis suggests there is no single “risky road”
          pattern. Residential persistent-risk segments and elevated corridor
          segments appear to reflect different structural environments and likely
          require different interventions.
        </p>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
        <h3 className="text-lg font-semibold">Risk changes over time</h3>
        <p className="mt-3 text-sm leading-6 text-slate-600">
          The HMM shows that citywide crash risk shifts between stable baseline
          conditions and more persistent elevated periods. That means
          intervention timing matters, not just location.
        </p>
      </div>
    </div>
  );
}