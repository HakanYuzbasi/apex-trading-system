"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/lib/auth-context";

interface AdvancedMetrics {
  available: boolean;
  note?: string;
  n_returns?: number;
  cvar_95?: number | null;
  cvar_99?: number | null;
  var_95?: number | null;
  var_99?: number | null;
  sortino_ratio?: number | null;
  calmar_ratio?: number | null;
  omega_ratio?: number | null;
  downside_deviation?: number | null;
  tail_ratio?: number | null;
  skewness?: number | null;
  kurtosis?: number | null;
  max_dd_duration?: number | null;
}

function fmt(v: number | null | undefined, decimals = 3): string {
  if (v == null || !isFinite(v)) return "—";
  return v.toFixed(decimals);
}

function pct(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return "—";
  return (v * 100).toFixed(2) + "%";
}

function MetricCard({
  label,
  value,
  subtitle,
  color,
}: {
  label: string;
  value: string;
  subtitle?: string;
  color?: "green" | "red" | "yellow" | "blue";
}) {
  const colorClass =
    color === "green"
      ? "text-green-400"
      : color === "red"
      ? "text-red-400"
      : color === "yellow"
      ? "text-yellow-400"
      : color === "blue"
      ? "text-blue-400"
      : "text-foreground";
  return (
    <div className="flex flex-col gap-0.5 rounded-lg border border-border/60 bg-background/50 px-3 py-2.5">
      <span className="text-[11px] text-muted-foreground uppercase tracking-wide">{label}</span>
      <span className={`text-base font-semibold font-mono ${colorClass}`}>{value}</span>
      {subtitle && <span className="text-[10px] text-muted-foreground/60">{subtitle}</span>}
    </div>
  );
}

export default function AdvancedMetricsPanel() {
  const { token } = useAuthContext();
  const [data, setData] = useState<AdvancedMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch("/api/v1/advanced-metrics", {
          headers: token ? { authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const id = setInterval(load, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [token]);

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading advanced metrics…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "Advanced metrics unavailable — engine may not be running or insufficient history."}
      </div>
    );
  }

  const sortinoColor = (v?: number | null) => {
    if (v == null) return undefined;
    return v >= 1.5 ? "green" : v >= 0.5 ? "yellow" : "red";
  };
  const calmarColor = (v?: number | null) => {
    if (v == null) return undefined;
    return v >= 0.5 ? "green" : v >= 0.2 ? "yellow" : "red";
  };
  const cvarColor = (v?: number | null) => {
    if (v == null) return undefined;
    return v >= -0.02 ? "green" : v >= -0.05 ? "yellow" : "red";
  };

  return (
    <div className="p-4 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-foreground">Advanced Risk Metrics</h2>
        <span className="text-[11px] text-muted-foreground font-mono">{data.n_returns ?? 0} returns</span>
      </div>

      {/* Tail Risk */}
      <section>
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Tail Risk</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <MetricCard label="CVaR 95%" value={pct(data.cvar_95)} subtitle="Expected Shortfall" color={cvarColor(data.cvar_95)} />
          <MetricCard label="CVaR 99%" value={pct(data.cvar_99)} subtitle="1% tail loss" color={cvarColor(data.cvar_99)} />
          <MetricCard label="VaR 95%" value={pct(data.var_95)} subtitle="1-day at 95%" color={cvarColor(data.var_95)} />
          <MetricCard label="VaR 99%" value={pct(data.var_99)} subtitle="1-day at 99%" color={cvarColor(data.var_99)} />
        </div>
      </section>

      {/* Risk-Adjusted Returns */}
      <section>
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Risk-Adjusted Returns</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          <MetricCard label="Sortino Ratio" value={fmt(data.sortino_ratio, 2)} subtitle="Downside-adjusted return" color={sortinoColor(data.sortino_ratio)} />
          <MetricCard label="Calmar Ratio" value={fmt(data.calmar_ratio, 2)} subtitle="Return / Max Drawdown" color={calmarColor(data.calmar_ratio)} />
          <MetricCard label="Omega Ratio" value={fmt(data.omega_ratio, 2)} subtitle="Gain / Loss probability" color={data.omega_ratio != null && data.omega_ratio > 1 ? "green" : "red"} />
        </div>
      </section>

      {/* Distribution */}
      <section>
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Return Distribution</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <MetricCard label="Downside σ" value={pct(data.downside_deviation)} subtitle="Downside deviation" />
          <MetricCard
            label="Tail Ratio"
            value={fmt(data.tail_ratio, 2)}
            subtitle="95th / 5th pct"
            color={data.tail_ratio != null && data.tail_ratio > 1 ? "green" : "yellow"}
          />
          <MetricCard
            label="Skewness"
            value={fmt(data.skewness, 3)}
            subtitle={data.skewness != null && data.skewness < 0 ? "Negative (left tail)" : "Positive (right tail)"}
            color={data.skewness != null && data.skewness >= 0 ? "green" : "yellow"}
          />
          <MetricCard
            label="Kurtosis"
            value={fmt(data.kurtosis, 2)}
            subtitle={data.kurtosis != null && data.kurtosis > 3 ? "Fat tails (>3)" : "Normal tails"}
            color={data.kurtosis != null && data.kurtosis > 5 ? "red" : "blue"}
          />
        </div>
      </section>

      {/* Drawdown Duration */}
      {data.max_dd_duration != null && (
        <section>
          <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2.5 flex items-center justify-between">
            <div>
              <span className="text-[11px] text-muted-foreground uppercase tracking-wide">Max Drawdown Duration</span>
              <p className="text-sm text-muted-foreground mt-0.5">
                Longest streak of consecutive losses (bars below previous peak)
              </p>
            </div>
            <span className={`text-xl font-bold font-mono ${data.max_dd_duration > 20 ? "text-red-400" : data.max_dd_duration > 10 ? "text-yellow-400" : "text-green-400"}`}>
              {data.max_dd_duration}
              <span className="text-xs text-muted-foreground ml-1">bars</span>
            </span>
          </div>
        </section>
      )}
    </div>
  );
}
