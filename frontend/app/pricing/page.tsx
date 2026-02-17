"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type PlanTier = "free" | "basic" | "pro" | "enterprise";

type PlanOffer = {
  code: string;
  name: string;
  tier: PlanTier;
  monthly_usd: number;
  annual_usd: number;
  recommended: boolean;
  target_user: string;
  usp: string;
  feature_highlights: string[];
  feature_limits: Record<string, number>;
};

const TIER_ORDER: Record<PlanTier, number> = {
  free: 0,
  basic: 1,
  pro: 2,
  enterprise: 3,
};

function formatLimit(value: number): string {
  if (value < 0) return "Unlimited";
  return `${value}/day`;
}

function formatPrice(plan: PlanOffer, annual: boolean): string {
  if (plan.monthly_usd === 0) return "Free";
  if (annual) return `$${plan.annual_usd.toLocaleString()}/yr`;
  return `$${plan.monthly_usd.toLocaleString()}/mo`;
}

export default function PricingPage() {
  const [plans, setPlans] = useState<PlanOffer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [annual, setAnnual] = useState(false);

  useEffect(() => {
    let active = true;
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const res = await fetch("/api/v1/plans", { cache: "no-store" });
        if (!res.ok) throw new Error("Unable to load pricing plans.");
        const payload = (await res.json()) as PlanOffer[];
        if (!active) return;
        setPlans(Array.isArray(payload) ? payload : []);
      } catch (err: unknown) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to load pricing.");
      } finally {
        if (active) setLoading(false);
      }
    };

    void load();
    return () => {
      active = false;
    };
  }, []);

  const sortedPlans = useMemo(
    () => [...plans].sort((a, b) => TIER_ORDER[a.tier] - TIER_ORDER[b.tier]),
    [plans],
  );

  const featureKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const plan of sortedPlans) {
      for (const key of Object.keys(plan.feature_limits || {})) {
        keys.add(key);
      }
    }
    return Array.from(keys).sort();
  }, [sortedPlans]);

  return (
    <main className="apex-shell min-h-screen px-4 py-8 sm:px-6 lg:px-10">
      <div className="mx-auto w-full max-w-7xl space-y-6">
        <section className="apex-panel rounded-3xl border border-border/80 p-6 sm:p-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-3">
              <p className="inline-flex items-center rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold tracking-wide text-primary">
                APEX Pricing
              </p>
              <h1 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
                Hedge-fund-grade automation with a clear upgrade path
              </h1>
              <p className="max-w-3xl text-sm text-muted-foreground sm:text-base">
                Unique selling point: adaptive governor policy tuning + execution shield + attribution loop in one PM cockpit.
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <Link
                  href="/login"
                  className="rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-foreground transition hover:bg-muted/50"
                >
                  Back to Login
                </Link>
              </div>
            </div>

            <div className="apex-segment" role="group" aria-label="Billing period toggle">
              <button
                type="button"
                className={`apex-segment-button ${!annual ? "is-active" : ""}`}
                onClick={() => setAnnual(false)}
              >
                Monthly
              </button>
              <button
                type="button"
                className={`apex-segment-button ${annual ? "is-active" : ""}`}
                onClick={() => setAnnual(true)}
              >
                Annual
              </button>
            </div>
          </div>
        </section>

        {error ? (
          <section className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </section>
        ) : null}

        <section className="grid gap-4 lg:grid-cols-4">
          {loading
            ? Array.from({ length: 4 }).map((_, idx) => (
                <article key={`loading-${idx}`} className="apex-panel rounded-2xl border border-border/70 p-5">
                  <div className="h-4 w-24 animate-pulse rounded bg-muted" />
                  <div className="mt-3 h-8 w-36 animate-pulse rounded bg-muted" />
                  <div className="mt-3 h-3 w-full animate-pulse rounded bg-muted" />
                  <div className="mt-5 h-20 w-full animate-pulse rounded bg-muted" />
                </article>
              ))
            : sortedPlans.map((plan) => (
                <article
                  key={plan.code}
                  className={`apex-panel apex-interactive rounded-2xl border p-5 ${
                    plan.recommended
                      ? "border-primary/70 bg-primary/5"
                      : "border-border/75"
                  }`}
                >
                  <div className="mb-3 flex items-start justify-between gap-2">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">{plan.tier}</p>
                      <h2 className="text-xl font-semibold text-foreground">{plan.name}</h2>
                    </div>
                    {plan.recommended ? (
                      <span className="rounded-full bg-emerald-100 px-2 py-1 text-[11px] font-semibold text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300">
                        Recommended
                      </span>
                    ) : null}
                  </div>

                  <p className="text-sm text-muted-foreground">{plan.target_user}</p>
                  <p className="mt-2 text-sm text-foreground">{plan.usp}</p>
                  <p className="apex-kpi-value mt-4 text-3xl font-semibold text-foreground">
                    {formatPrice(plan, annual)}
                  </p>
                  {plan.monthly_usd > 0 && annual ? (
                    <p className="text-xs text-muted-foreground">
                      Equivalent to ${(plan.annual_usd / 12).toFixed(0)}/mo
                    </p>
                  ) : null}

                  <ul className="mt-4 space-y-1.5 text-sm text-muted-foreground">
                    {plan.feature_highlights.slice(0, 3).map((item) => (
                      <li key={item}>- {item}</li>
                    ))}
                  </ul>

                  <div className="mt-5 flex items-center gap-2">
                    <Link
                      href="/login"
                      className="rounded-lg bg-primary px-3 py-2 text-xs font-semibold text-primary-foreground transition hover:opacity-90"
                    >
                      Start with {plan.name}
                    </Link>
                    <Link
                      href="/settings"
                      className="rounded-lg border border-border px-3 py-2 text-xs font-semibold text-foreground transition hover:bg-muted/50"
                    >
                      Manage Plan
                    </Link>
                  </div>
                </article>
              ))}
        </section>

        <section className="apex-panel rounded-2xl border border-border/80 p-5">
          <h3 className="text-lg font-semibold text-foreground">Feature Matrix</h3>
          <p className="mb-4 text-sm text-muted-foreground">
            Plan limits are sourced from the same backend catalog used by in-app subscription settings.
          </p>

          <div className="overflow-auto rounded-xl border border-border/70">
            <table className="min-w-full text-sm">
              <thead className="bg-background/95 text-left text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-semibold">Feature</th>
                  {sortedPlans.map((plan) => (
                    <th key={plan.code} className="px-3 py-2 font-semibold">
                      {plan.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {featureKeys.length === 0 ? (
                  <tr>
                    <td colSpan={Math.max(2, sortedPlans.length + 1)} className="px-3 py-6 text-center text-muted-foreground">
                      No feature limits returned yet.
                    </td>
                  </tr>
                ) : (
                  featureKeys.map((feature) => (
                    <tr key={feature} className="border-t border-border/60">
                      <td className="px-3 py-2 text-foreground">{feature}</td>
                      {sortedPlans.map((plan) => {
                        const raw = plan.feature_limits?.[feature];
                        return (
                          <td key={`${plan.code}-${feature}`} className="px-3 py-2 text-muted-foreground">
                            {typeof raw === "number" ? formatLimit(raw) : "-"}
                          </td>
                        );
                      })}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}
