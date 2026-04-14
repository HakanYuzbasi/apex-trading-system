"use client";

import { useEffect, useState, useCallback } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";
import { apiFetch } from "@/lib/api";
import { Activity, RefreshCw } from "lucide-react";

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
  profit_factor?: number | null;
  expectancy?: number | null;
  health?: {
    rolling_sharpe: number;
    trade_count: number;
    paper_only: boolean;
    reason: string;
    last_updated: string;
  };
  model_drift?: {
    health: string;
    ic_current: number;
    hit_rate_current: number;
    med_confidence: number;
  };
  signal_quality?: {
    overall_score: number;
    hit_rate: number;
    avg_gain: number;
    avg_loss: number;
  };
}

function fmt(v: number | null | undefined, decimals = 3): string {
  if (v == null || !isFinite(Number(v))) return "—";
  return Number(v).toFixed(decimals);
}

function pct(v: number | null | undefined): string {
  if (v == null || !isFinite(Number(v))) return "—";
  const num = Number(v);
  if (Math.abs(num) < 0.00005) return "0.00%";
  return (num * 100).toFixed(2) + "%";
}

function MetricSkeleton() {
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-border/60 bg-background/50 px-3 py-2.5 animate-pulse">
      <div className="h-2.5 w-16 bg-muted rounded" />
      <div className="h-5 w-24 bg-muted/60 rounded mt-1" />
      <div className="h-2 w-20 bg-muted/40 rounded mt-1" />
    </div>
  );
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
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<AdvancedMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (isManual = false) => {
    if (isManual) setLoading(true);
    let cancelled = false;
    try {
      const res = await apiFetch("/api/v1/advanced-metrics", {
        cache: "no-store",
        token,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (!cancelled) {
        setData(json);
        setError(null);
      }
    } catch (e) {
      if (!cancelled) setError(String(e));
    } finally {
      if (!cancelled) setLoading(false);
    }
    return () => { cancelled = true; };
  }, [token]);

  useEffect(() => {
    load();
    const id = setInterval(() => load(), 5_000); // Increased 12x frequency
    return () => clearInterval(id);
  }, [load]);

  if (loading && !data) {
    return (
      <div className="p-4 space-y-5 min-h-[500px]">
        <div className="flex items-center justify-between mb-2">
          <div className="h-4 w-32 bg-muted rounded animate-pulse" />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {Array.from({ length: 8 }).map((_, i) => <MetricSkeleton key={i} />)}
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
          <div className="h-20 bg-muted/40 rounded-lg animate-pulse" />
          <div className="h-20 bg-muted/40 rounded-lg animate-pulse" />
        </div>
      </div>
    );
  }
  
  if (error && !data) return <div className="p-6 text-negative text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm flex items-center gap-2">
        <Activity className="h-4 w-4 animate-pulse text-warning" />
        {data?.note ?? "Institutional analytics engine initializing..."}
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
    <div className="p-4 space-y-5 min-h-[500px]">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-base font-semibold text-foreground">Advanced Risk Metrics</h2>
          <button 
            onClick={() => load(true)} 
            disabled={loading}
            className="p-1 hover:bg-muted rounded-md transition-colors"
          >
            <RefreshCw className={`h-3.5 w-3.5 text-muted-foreground ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
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
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Institutional Analytics</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <MetricCard label="Sortino Ratio" value={fmt(data.sortino_ratio, 2)} subtitle="Downside efficiency" color={sortinoColor(data.sortino_ratio)} />
          <MetricCard label="Calmar Ratio" value={fmt(data.calmar_ratio, 2)} subtitle="Return/MDD" color={calmarColor(data.calmar_ratio)} />
          <MetricCard label="Profit Factor" value={fmt(data.profit_factor, 2)} subtitle="Gross Win / Gross Loss" color={data.profit_factor != null && data.profit_factor > 1.2 ? "green" : "red"} />
          <MetricCard label="Expectancy" value={pct(data.expectancy)} subtitle="Avg return per trade" color={data.expectancy != null && data.expectancy > 0 ? "green" : "red"} />
        </div>
      </section>

      {/* Model Health Overlays */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {data.health && (
          <section className="rounded-lg border border-border/40 bg-secondary/10 p-3">
            <h3 className="text-[10px] font-bold text-muted-foreground uppercase mb-2">Strategy Health (30D)</h3>
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-lg font-bold ${data.health.paper_only ? "text-negative" : "text-positive"}`}>
                  {fmt(data.health.rolling_sharpe, 2)}
                  <span className="text-xs font-normal text-muted-foreground ml-1">Sharpe</span>
                </p>
                <p className="text-[10px] text-muted-foreground">{data.health.reason}</p>
              </div>
              <div className="text-right">
                <span className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase ${data.health.paper_only ? "bg-negative/20 text-negative" : "bg-positive/20 text-positive"}`}>
                  {data.health.paper_only ? "Paper Only" : "Active"}
                </span>
                <p className="text-[9px] text-muted-foreground mt-1">{data.health.trade_count} trades</p>
              </div>
            </div>
          </section>
        )}

        {data.model_drift && (
          <section className="rounded-lg border border-border/40 bg-secondary/10 p-3">
            <h3 className="text-[10px] font-bold text-muted-foreground uppercase mb-2">Model Drift index</h3>
            <div className="flex items-center justify-between">
              <div>
                <p className={`text-lg font-bold ${data.model_drift.health === "healthy" ? "text-positive" : "text-warning"}`}>
                  {fmt(data.model_drift.ic_current * 10, 2)}
                  <span className="text-xs font-normal text-muted-foreground ml-1">IC Score</span>
                </p>
                <p className="text-[10px] text-muted-foreground">Hit Rate: {(Number(data.model_drift.hit_rate_current) * 100).toFixed(1)}%</p>
              </div>
              <div className="text-right">
                <span className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase ${data.model_drift.health === "healthy" ? "bg-positive/20 text-positive" : "bg-warning/20 text-warning"}`}>
                  {data.model_drift.health}
                </span>
                <p className="text-[9px] text-muted-foreground mt-1">Confidence: {fmt(data.model_drift.med_confidence, 2)}</p>
              </div>
            </div>
          </section>
        )}
      </div>

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
