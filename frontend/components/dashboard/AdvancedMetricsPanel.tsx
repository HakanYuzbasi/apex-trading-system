"use client";

import { useEffect, useState, useCallback } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";
import { apiFetch } from "@/lib/api";
import { Activity, RefreshCw, ShieldCheck, Zap, AlertTriangle, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { getToneClass } from "@/lib/utils";

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
  if (v == null || isNaN(Number(v)) || !isFinite(Number(v))) return "—";
  return Number(v).toFixed(decimals);
}

function pct(v: number | null | undefined): string {
  if (v == null || isNaN(Number(v)) || !isFinite(Number(v))) return "—";
  const num = Number(v);
  if (Math.abs(num) < 0.00005) return "0.00%";
  return (num * 100).toFixed(2) + "%";
}

function MetricSkeleton() {
  return (
    <div className="flex flex-col gap-1.5 rounded-xl border border-border/10 bg-background/20 px-3 py-3 animate-pulse">
      <div className="h-2 w-12 bg-muted/40 rounded" />
      <div className="h-4 w-20 bg-muted/30 rounded mt-1" />
    </div>
  );
}

function MetricCard({
  label,
  value,
  subtitle,
  tone = "neutral",
}: {
  label: string;
  value: string;
  subtitle?: string;
  tone?: "positive" | "negative" | "warning" | "neutral";
}) {
  return (
    <div className="flex flex-col gap-1 rounded-xl border border-border/10 bg-background/30 px-3 py-3 transition-all hover:bg-background/50 hover:border-primary/20 hover:scale-[1.02] cursor-default shadow-sm hover:shadow-md duration-300">
      <span className="text-[9px] font-black text-muted-foreground uppercase tracking-widest leading-none">{label}</span>
      <span className={`text-base font-black font-mono leading-none py-1 ${getToneClass(tone, "text")}`}>
        {value}
      </span>
      {subtitle && <span className="text-[9px] font-medium text-muted-foreground/60 truncate tracking-tight">{subtitle}</span>}
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
    const id = setInterval(() => load(), 5_000);
    return () => clearInterval(id);
  }, [load]);

  if (loading && !data) {
    return (
      <div className="glass-card rounded-2xl p-5 space-y-6 min-h-[500px]">
        <div className="flex items-center justify-between border-b border-border/20 pb-4">
           <div className="h-4 w-40 bg-muted/30 rounded-lg animate-pulse" />
           <div className="h-5 w-16 bg-muted/20 rounded-full animate-pulse" />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {Array.from({ length: 8 }).map((_, i) => <MetricSkeleton key={i} />)}
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
          <div className="h-24 bg-muted/20 rounded-xl animate-pulse" />
          <div className="h-24 bg-muted/20 rounded-xl animate-pulse" />
        </div>
      </div>
    );
  }
  
  if (error && !data) {
    return (
      <div className="glass-card rounded-2xl p-8 text-center space-y-3">
        <AlertTriangle className="h-8 w-8 text-negative mx-auto" />
        <p className="text-xs font-black text-negative uppercase tracking-widest">Metrics Telemetry Failed</p>
        <p className="text-[10px] text-muted-foreground max-w-sm mx-auto">{error}</p>
        <button onClick={() => load(true)} className="px-4 py-2 bg-negative/10 text-negative text-[10px] font-black uppercase rounded-lg hover:bg-negative/20">RETRY SYNC</button>
      </div>
    );
  }

  if (!data || !data.available) {
    return (
      <div className="glass-card rounded-2xl p-10 flex flex-col items-center justify-center text-center space-y-4">
        <Activity className="h-10 w-10 animate-pulse text-primary opacity-50" />
        <div className="space-y-1">
           <p className="text-[10px] font-black text-foreground uppercase tracking-[0.2em]">Institutional Analytics engine</p>
           <p className="text-xs text-muted-foreground">{data?.note ?? "Synchronizing real-time risk headers..."}</p>
        </div>
      </div>
    );
  }

  const getSortinoTone = (v?: number | null) => (v == null ? "neutral" : v >= 1.5 ? "positive" : v >= 0.5 ? "warning" : "negative");
  const getCalmarTone = (v?: number | null) => (v == null ? "neutral" : v >= 0.5 ? "positive" : v >= 0.2 ? "warning" : "negative");
  const getCvarTone = (v?: number | null) => {
    if (v == null) return "neutral";
    const num = Number(v);
    // CVaR is typically negative; a "more negative" value is higher risk (negative tone)
    return num <= -0.05 ? "negative" : num <= -0.02 ? "warning" : "positive";
  };

  return (
    <div className="glass-card rounded-2xl p-6 space-y-8 min-h-[500px] animate-in fade-in duration-700">
      <div className="flex items-center justify-between border-b border-border/20 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 border border-primary/20 shadow-[0_0_15px_rgba(var(--primary-rgb),0.1)]">
             <ShieldCheck className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h2 className="text-[11px] font-black text-foreground uppercase tracking-widest">INSTITUTIONAL RISK OVERLAY</h2>
            <div className="flex items-center gap-2 mt-0.5">
               <span className="text-[9px] font-black text-muted-foreground/60 uppercase tracking-tighter tabular-nums">{data.n_returns ?? 0} RETURNS SAMPLED</span>
               <span className="h-1 w-1 rounded-full bg-border" />
               <button 
                onClick={() => load(true)} 
                disabled={loading}
                className="group flex items-center gap-1.5 hover:text-primary transition-colors"
              >
                <RefreshCw className={`h-2.5 w-2.5 ${loading ? "animate-spin" : "group-hover:rotate-180 transition-transform duration-500"}`} />
                <span className="text-[9px] font-black text-muted-foreground/80 uppercase tracking-tighter">Sync</span>
              </button>
            </div>
          </div>
        </div>
        <Badge variant="outline" className="text-[9px] font-bold bg-background/40 py-1 tracking-widest">LIVE ANALYTICS</Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* Tail Risk Column */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 px-1 border-l-2 border-primary/40">
            <TrendingUp size={12} className="text-primary" />
            <h3 className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.15em]">TAIL RISK</h3>
          </div>
          <div className="grid grid-cols-2 gap-2.5">
            <MetricCard label="CVaR 95%" value={pct(data.cvar_95)} subtitle="Expected Shortfall" tone={getCvarTone(data.cvar_95)} />
            <MetricCard label="CVaR 99%" value={pct(data.cvar_99)} subtitle="1% Tail Loss" tone={getCvarTone(data.cvar_99)} />
            <MetricCard label="VaR 95%" value={pct(data.var_95)} subtitle="1-Day at 95%" tone={getCvarTone(data.var_95)} />
            <MetricCard label="VaR 99%" value={pct(data.var_99)} subtitle="1-Day at 99%" tone={getCvarTone(data.var_99)} />
          </div>
        </div>

        {/* Efficiency Column */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 px-1 border-l-2 border-warning/40">
            <Zap size={12} className="text-warning" />
            <h3 className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.15em]">PERFORMANCE EFFICIENCY</h3>
          </div>
          <div className="grid grid-cols-2 gap-2.5">
            <MetricCard label="Sortino Ratio" value={fmt(data.sortino_ratio, 2)} subtitle="Downside efficiency" tone={getSortinoTone(data.sortino_ratio)} />
            <MetricCard label="Calmar Ratio" value={fmt(data.calmar_ratio, 2)} subtitle="Return/MDD" tone={getCalmarTone(data.calmar_ratio)} />
            <MetricCard label="Profit Factor" value={fmt(data.profit_factor, 2)} subtitle="Gross Gain/Loss" tone={data.profit_factor != null && data.profit_factor > 1.2 ? "positive" : "negative"} />
            <MetricCard label="Expectancy" value={pct(data.expectancy)} subtitle="Avg return / trade" tone={data.expectancy != null && data.expectancy > 0 ? "positive" : "negative"} />
          </div>
        </div>

        {/* Distribution Column */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 px-1 border-l-2 border-muted-foreground/40">
            <Activity size={12} className="text-muted-foreground" />
            <h3 className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.15em]">RETURN DISTRIBUTION</h3>
          </div>
          <div className="grid grid-cols-2 gap-2.5">
            <MetricCard label="Downside σ" value={pct(data.downside_deviation)} subtitle="Downside Stdev" />
            <MetricCard label="Tail Ratio" value={fmt(data.tail_ratio, 2)} subtitle="95th / 5th pct" tone={data.tail_ratio != null && data.tail_ratio > 1 ? "positive" : "warning"} />
            <MetricCard label="Skewness" value={fmt(data.skewness, 3)} subtitle={data.skewness != null && data.skewness < 0 ? "Left tail leak" : "Positive skew"} tone={data.skewness != null && data.skewness >= 0 ? "positive" : "warning"} />
            <MetricCard label="Kurtosis" value={fmt(data.kurtosis, 2)} subtitle={data.kurtosis != null && data.kurtosis > 3 ? "Fat tails (>3)" : "Normal dist"} tone={data.kurtosis != null && data.kurtosis > 5 ? "negative" : "neutral"} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {data.health && (
          <div className="p-4 rounded-xl border border-border/10 bg-background/30 flex items-center justify-between transition-all hover:bg-background/50 hover:shadow-lg duration-300">
            <div className="space-y-1">
              <span className="text-[9px] font-black text-muted-foreground uppercase tracking-widest opacity-50">STRATEGY HEALTH (30D)</span>
              <p className={`text-xl font-black font-mono leading-none ${getToneClass(data.health.paper_only ? "negative" : "positive", "text")}`}>
                {fmt(data.health.rolling_sharpe, 2)}
                <span className="text-[10px] font-black text-muted-foreground ml-2 opacity-60">ROLLING SHARPE</span>
              </p>
              <p className="text-[10px] font-medium text-muted-foreground/80">{data.health.reason}</p>
            </div>
            <div className="text-right space-y-2">
               <Badge variant={data.health.paper_only ? "destructive" : "default"} className="text-[9px] font-black uppercase py-0.5">
                  {data.health.paper_only ? "Paper Core" : "Active Core"}
               </Badge>
               <p className="text-[10px] font-black font-mono text-muted-foreground">{data.health.trade_count} TRADES</p>
            </div>
          </div>
        )}

        {data.model_drift && (
          <div className="p-4 rounded-xl border border-border/10 bg-background/30 flex items-center justify-between transition-all hover:bg-background/50 hover:shadow-lg duration-300">
            <div className="space-y-1">
              <span className="text-[9px] font-black text-muted-foreground uppercase tracking-widest opacity-50">MODEL DRIFT AUDIT</span>
              <p className={`text-xl font-black font-mono leading-none ${getToneClass(data.model_drift.health === "healthy" ? "positive" : "warning", "text")}`}>
                {fmt(data.model_drift.ic_current * 10, 2)}
                <span className="text-[10px] font-black text-muted-foreground ml-2 opacity-60">IC SCORE</span>
              </p>
              <p className="text-[10px] font-medium text-muted-foreground/80 lowercase">Hit rate: {pct(data.model_drift.hit_rate_current / 100)}</p>
            </div>
            <div className="text-right space-y-2">
               <Badge variant="outline" className={`text-[9px] font-black uppercase py-0.5 ${getToneClass(data.model_drift.health === "healthy" ? "positive" : "warning", "bg")}`}>
                  {data.model_drift.health}
               </Badge>
               <p className="text-[10px] font-black font-mono text-muted-foreground uppercase">CONF: {fmt(data.model_drift.med_confidence, 2)}</p>
            </div>
          </div>
        )}
      </div>

      {data.max_dd_duration != null && (
        <div className="rounded-xl border border-border/10 bg-background/40 p-5 flex items-center justify-between shadow-inner">
          <div className="space-y-1">
            <h4 className="text-[10px] font-black text-foreground uppercase tracking-widest">Max Drawdown Duration</h4>
            <p className="text-[10px] text-muted-foreground max-w-md">Longest continuous streak of negative equity bars relative to the previous high-water mark.</p>
          </div>
          <div className="flex items-baseline gap-1.5">
             <span className={`text-2xl font-black font-mono ${data.max_dd_duration > 20 ? "text-negative" : data.max_dd_duration > 10 ? "text-warning" : "text-positive"}`}>
                {data.max_dd_duration}
             </span>
             <span className="text-[10px] font-black text-muted-foreground/60 uppercase">BARS</span>
          </div>
        </div>
      )}
    </div>
  );
}
