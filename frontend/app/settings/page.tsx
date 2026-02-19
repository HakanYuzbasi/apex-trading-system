"use client";

import { useEffect, useMemo, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";
import Link from "next/link";
import BrokerConnections from "@/components/BrokerConnections";

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

const TIER_RANK: Record<PlanTier, number> = {
  free: 0,
  basic: 1,
  pro: 2,
  enterprise: 3,
};

export default function SettingsPage() {
  const { user, logout, isAuthenticated, accessToken } = useAuthContext();
  const [plans, setPlans] = useState<PlanOffer[]>([]);
  const [billingError, setBillingError] = useState<string>("");
  const [loadingPlans, setLoadingPlans] = useState<boolean>(true);
  const [checkoutTier, setCheckoutTier] = useState<PlanTier | null>(null);

  useEffect(() => {
    let active = true;
    const fetchPlans = async () => {
      setLoadingPlans(true);
      setBillingError("");
      try {
        const res = await fetch("/api/v1/plans", { cache: "no-store" });
        if (!res.ok) {
          throw new Error("Failed to load subscription plans.");
        }
        const payload = (await res.json()) as PlanOffer[];
        if (!active) return;
        setPlans(Array.isArray(payload) ? payload : []);
      } catch (error: unknown) {
        if (!active) return;
        const detail = error instanceof Error ? error.message : "Unable to load plan catalog.";
        setBillingError(detail);
      } finally {
        if (active) {
          setLoadingPlans(false);
        }
      }
    };

    void fetchPlans();
    return () => {
      active = false;
    };
  }, []);

  const sortedPlans = useMemo(() => {
    return [...plans].sort((a, b) => TIER_RANK[a.tier] - TIER_RANK[b.tier]);
  }, [plans]);

  const handleCheckout = async (tier: PlanTier) => {
    if (!accessToken) {
      setBillingError("You must be signed in to start checkout.");
      return;
    }
    setCheckoutTier(tier);
    setBillingError("");
    try {
      const base = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
      const response = await fetch(`${base}/auth/billing/checkout`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ tier }),
      });
      const payload = (await response.json().catch(() => ({}))) as { checkout_url?: string; detail?: string };
      if (!response.ok || !payload.checkout_url) {
        throw new Error(payload.detail || "Checkout is currently unavailable.");
      }
      window.location.href = payload.checkout_url;
    } catch (error: unknown) {
      const detail = error instanceof Error ? error.message : "Checkout failed.";
      setBillingError(detail);
    } finally {
      setCheckoutTier(null);
    }
  };

  if (!isAuthenticated || !user) {
    return (
      <main className="min-h-screen flex flex-col items-center justify-center p-4">
        <p className="text-muted-foreground">You must be signed in to view settings.</p>
        <Link href="/login" className="mt-2 text-primary hover:underline">Sign in</Link>
      </main>
    );
  }

  return (
    <main className="apex-shell min-h-screen px-4 py-8 sm:px-6 lg:px-10">
      <div className="mx-auto w-full max-w-6xl space-y-6">
        <section className="apex-panel rounded-3xl border border-border/80 p-6 sm:p-8">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-3xl font-semibold tracking-tight text-foreground">Account & Subscription</h1>
              <p className="mt-1 text-sm text-muted-foreground">
                Keep your trading stack and plan entitlements aligned.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Link
                href="/pricing"
                className="rounded-lg border border-border px-3 py-2 text-xs font-semibold text-foreground transition hover:bg-muted/50"
              >
                View Public Pricing
              </Link>
              <Link
                href="/dashboard"
                className="rounded-lg bg-primary px-3 py-2 text-xs font-semibold text-primary-foreground transition hover:opacity-90"
              >
                Back to Dashboard
              </Link>
            </div>
          </div>
        </section>

        <section className="grid gap-4 md:grid-cols-[1fr_auto]">
          <article className="apex-panel rounded-2xl border border-border/75 p-4">
            <p><span className="text-muted-foreground">Username:</span> {user.username}</p>
            {user.email && <p className="mt-1"><span className="text-muted-foreground">Email:</span> {user.email}</p>}
            <p className="mt-1"><span className="text-muted-foreground">Tier:</span> <span className="capitalize">{user.tier}</span></p>
          </article>
          <article className="apex-panel rounded-2xl border border-border/75 p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Operator actions</p>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                onClick={() => logout()}
                className="rounded-lg border border-border px-4 py-2 text-foreground hover:bg-muted/50 transition"
              >
                Sign out
              </button>
            </div>
          </article>
        </section>

        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-foreground">Plan lineup</h2>
            <p className="text-xs text-muted-foreground">Unique selling point: adaptive governor + execution shield + attribution loop</p>
          </div>
          {billingError ? (
            <p className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">{billingError}</p>
          ) : null}
          {loadingPlans ? (
            <p className="text-sm text-muted-foreground">Loading plans...</p>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              {sortedPlans.map((plan) => {
                const isCurrent = plan.tier === user.tier;
                const isUpgrade = TIER_RANK[plan.tier] > TIER_RANK[user.tier];
                const ctaDisabled = !isUpgrade || checkoutTier === plan.tier;
                return (
                  <article
                    key={plan.code}
                    className={`rounded-xl border p-4 transition ${isCurrent
                      ? "border-primary bg-primary/10"
                      : "border-border bg-card/60 hover:border-primary/40"
                      }`}
                  >
                    <div className="mb-3 flex items-start justify-between gap-2">
                      <div>
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">{plan.tier}</p>
                        <h3 className="text-lg font-semibold text-foreground">{plan.name}</h3>
                      </div>
                      {plan.recommended ? (
                        <span className="rounded-full bg-emerald-100 px-2 py-1 text-[11px] font-semibold text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300">
                          Recommended
                        </span>
                      ) : null}
                    </div>
                    <p className="text-sm text-muted-foreground">{plan.target_user}</p>
                    <p className="mt-2 text-sm text-foreground">{plan.usp}</p>
                    <p className="mt-3 text-2xl font-semibold text-foreground">
                      {plan.monthly_usd === 0 ? "Free" : `$${plan.monthly_usd.toLocaleString()}/mo`}
                    </p>
                    {plan.annual_usd > 0 ? (
                      <p className="text-xs text-muted-foreground">${plan.annual_usd.toLocaleString()} billed annually</p>
                    ) : null}
                    <ul className="mt-3 space-y-1 text-sm text-muted-foreground">
                      {plan.feature_highlights.slice(0, 3).map((item) => (
                        <li key={item}>- {item}</li>
                      ))}
                    </ul>
                    <div className="mt-4 flex items-center gap-2">
                      {isCurrent ? (
                        <span className="rounded-md border border-primary/50 bg-primary/15 px-3 py-1.5 text-xs font-semibold text-primary">
                          Current Plan
                        </span>
                      ) : (
                        <button
                          type="button"
                          disabled={ctaDisabled}
                          onClick={() => handleCheckout(plan.tier)}
                          className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${ctaDisabled
                            ? "cursor-not-allowed border border-border text-muted-foreground"
                            : "bg-primary text-primary-foreground hover:opacity-90"
                            }`}
                        >
                          {checkoutTier === plan.tier ? "Opening checkout..." : isUpgrade ? "Upgrade" : "Included"}
                        </button>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        {/* ═══ Broker Connections ═══ */}
        <section className="apex-panel rounded-2xl border border-border/75 p-4 sm:p-6">
          <BrokerConnections accessToken={accessToken} />
        </section>

      </div>
    </main>
  );
}
