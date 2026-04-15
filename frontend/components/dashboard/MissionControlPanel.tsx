"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";
import { Activity, Shield, Zap, TrendingUp, TrendingDown, RefreshCw, BarChart3, AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn, getToneClass } from "@/lib/utils";

interface MissionData {
  system: {
    regime: string;
    vix: number;
    kill_switch_active: boolean;
    governor_tier: string;
    equity: number;
    meta_confidence: number;
    survival_probability: number;
  };
  risk_budget: {
    daily_pnl: number;
    daily_pnl_pct: number;
    realized_pnl: number;
    unrealized_pnl: number;
    drawdown_pct: number;
    active_margin: number;
    position_count: number;
    max_positions: number;
    positions_pct: number;
  };
  top_positions: {
    symbol: string;
    pnl_pct: number;
    pnl: number;
    qty: number;
    side: string;
    signal_direction: string;
  }[];
  predictive: {
    transition_probability: number | null;
    transition_direction: string | null;
    transition_size_mult: number | null;
    rl_epsilon: number | null;
    rl_total_updates: number | null;
    universe_scored: number | null;
    bayesian_vol_prob: number | null;
  };
  timestamp: string;
}

function tierVariant(tier: string): "positive" | "warning" | "negative" {
  if (tier === "RED") return "negative";
  if (tier === "YELLOW") return "warning";
  return "positive";
}

function regimeVariant(regime: string): "positive" | "warning" | "negative" {
  const r = regime.toLowerCase();
  if (r.includes("bear") || r.includes("crisis") || r.includes("volatile")) return "negative";
  if (r.includes("bull")) return "positive";
  return "warning";
}

function Bar({ pct, tone, label, value }: { pct: number; tone: "positive" | "warning" | "negative" | "neutral"; label: string; value: string }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const barClass = tone === "positive" ? "bg-positive" : tone === "warning" ? "bg-warning" : tone === "negative" ? "bg-negative" : "bg-primary";
  
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center px-1">
        <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{label}</span>
        <span className="text-[11px] font-mono font-bold text-foreground">{value}</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-background/40 border border-border/20 overflow-hidden">
        <div className={cn("h-full rounded-full transition-all duration-500", barClass)} style={{ width: `${clamped}%` }} />
      </div>
    </div>
  );
}

function ProbGauge({ prob }: { prob: number | null }) {
  if (prob === null) return <span className="text-muted-foreground text-[11px] font-bold">AWAITING_DATA</span>;
  const pct = Math.round(prob * 100);
  const color = pct >= 70 ? "bg-negative" : pct >= 45 ? "bg-warning" : "bg-positive";
  return (
    <div className="flex items-center gap-3">
      <div className="h-2 w-full rounded-full bg-background/40 border border-border/20 overflow-hidden">
        <div className={cn("h-full rounded-full transition-all duration-700", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-xs font-black min-w-[3ch]">{pct}%</span>
    </div>
  );
}

function Pill({ value, green, yellow }: { value: number; green: number; yellow: number }) {
  const variant = value >= green ? "positive" : value >= yellow ? "warning" : "negative";
  return (
    <Badge variant={variant} className="text-[10px] h-4.5 px-1.5 font-black bg-background/40">
      {(value * 100).toFixed(0)}%
    </Badge>
  );
}

export default function MissionControlPanel() {
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<MissionData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");

  const fetchData = async () => {
    try {
      const res = await fetch("/api/v1/mission-control", {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        cache: "no-store",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setLastUpdated(new Date().toLocaleTimeString());
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "fetch error");
    }
  };

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 10_000);
    return () => clearInterval(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  if (error)
    return (
      <div className="rounded-2xl border border-destructive/30 bg-destructive/10 p-6 flex items-center gap-3 animate-in shake duration-300">
        <AlertCircle className="h-5 w-5 text-destructive" />
        <span className="text-sm font-bold text-destructive">Mission Control Failure: {error}</span>
      </div>
    );

  if (!data)
    return (
      <div className="glass-card rounded-2xl p-12 flex flex-col items-center justify-center gap-4 text-muted-foreground animate-pulse">
        <RefreshCw className="h-8 w-8 animate-spin opacity-20" />
        <p className="text-xs font-bold uppercase tracking-widest">Acquiring Orbital State...</p>
      </div>
    );

  const { system, risk_budget, top_positions, predictive } = data;

  const dailyTone = risk_budget.daily_pnl >= 0 ? "positive" : risk_budget.daily_pnl_pct < -3 ? "negative" : "warning";
  const ddTone = risk_budget.drawdown_pct > 5 ? "negative" : risk_budget.drawdown_pct > 2 ? "warning" : "positive";
  const marginUsedPct = system.equity > 0 ? (risk_budget.active_margin / system.equity) * 100 : 0;
  const marginTone = marginUsedPct > 80 ? "negative" : marginUsedPct > 50 ? "warning" : "neutral";

  const fmtPnl = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
           <Shield className="h-5 w-5 text-primary" />
           <h2 className="text-xl font-bold text-foreground">Mission Control Center</h2>
        </div>
        <div className="flex items-center gap-3">
           <Badge variant="outline" className="text-[10px] font-bold bg-background/40">SYNC_AUTO: 10S</Badge>
           <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest whitespace-nowrap">T-REFRESH: {lastUpdated}</span>
        </div>
      </div>

      {/* ── Top Row: System Status + Risk Budget ── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">

        {/* System Status */}
        <div className="glass-card rounded-2xl p-6 flex flex-col justify-between">
          <div className="flex items-center gap-2 mb-6 border-b border-border/40 pb-4">
             <Activity className="h-4 w-4 text-primary" />
             <h3 className="text-xs font-black text-muted-foreground uppercase tracking-[0.2em]">Orbital System Status</h3>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
            <div className="space-y-1">
              <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Active Regime</p>
              <Badge variant={regimeVariant(system.regime)} className="h-6 px-3 text-[11px] font-black uppercase tracking-tight">
                {system.regime}
              </Badge>
            </div>
            <div className="space-y-1">
              <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">VIX Index</p>
              <p className={cn("text-lg font-black font-mono tracking-tighter", getToneClass(system.vix >= 30 ? "negative" : system.vix >= 20 ? "warning" : "positive"))}>
                {system.vix > 0 ? system.vix.toFixed(1) : "—"}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Governor Tier</p>
              <Badge variant={tierVariant(system.governor_tier)} className="h-6 px-3 text-[11px] font-black">
                {system.governor_tier}
              </Badge>
            </div>
            <div className="space-y-1">
              <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Kill Switch</p>
              <Badge variant={system.kill_switch_active ? "negative" : "positive"} className="h-6 px-3 text-[11px] font-black">
                {system.kill_switch_active ? "ENGAGED" : "OFF"}
              </Badge>
            </div>
          </div>

          <div className="mt-8 grid grid-cols-3 gap-4 pt-6 border-t border-border/40">
            <div className="space-y-1">
              <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Equity PNT</p>
              <p className="text-sm font-black font-mono text-foreground">${system.equity.toLocaleString()}</p>
            </div>
            <div className="space-y-1">
              <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Meta Confidence</p>
              <Pill value={system.meta_confidence} green={0.8} yellow={0.6} />
            </div>
            <div className="space-y-1">
              <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Survival Prob</p>
              <Pill value={system.survival_probability} green={0.9} yellow={0.7} />
            </div>
          </div>
        </div>

        {/* Risk Budget */}
        <div className="glass-card rounded-2xl p-6 space-y-5">
          <div className="flex items-center gap-2 mb-2 border-b border-border/40 pb-4">
             <Shield className="h-4 w-4 text-primary" />
             <h3 className="text-xs font-black text-muted-foreground uppercase tracking-[0.2em]">Risk Allocation Budget</h3>
          </div>

          <Bar
            pct={Math.abs(risk_budget.daily_pnl_pct) * 20}
            tone={dailyTone}
            label="Daily P&L Utilization"
            value={`${fmtPnl(risk_budget.daily_pnl)} (${risk_budget.daily_pnl >= 0 ? "+" : ""}${risk_budget.daily_pnl_pct.toFixed(2)}%)`}
          />
          <Bar
            pct={risk_budget.drawdown_pct * 10}
            tone={ddTone}
            label="Net Drawdown Consumption"
            value={`${risk_budget.drawdown_pct.toFixed(2)}%`}
          />
          <Bar
            pct={marginUsedPct}
            tone={marginTone}
            label="Active Margin Load"
            value={`$${risk_budget.active_margin.toLocaleString("en-US", { maximumFractionDigits: 0 })}`}
          />
          <Bar
            pct={risk_budget.positions_pct}
            tone={risk_budget.positions_pct >= 80 ? "warning" : "neutral"}
            label="Universe Slot Occupancy"
            value={`${risk_budget.position_count} / ${risk_budget.max_positions}`}
          />

          <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border/40">
            <div className="p-3 rounded-xl bg-background/20 border border-border/20">
              <p className="text-[9px] font-black text-muted-foreground uppercase tracking-widest mb-1">Realized</p>
              <p className={cn("text-xs font-black font-mono", getToneClass(risk_budget.realized_pnl >= 0 ? "positive" : "negative"))}>
                {fmtPnl(risk_budget.realized_pnl)}
              </p>
            </div>
            <div className="p-3 rounded-xl bg-background/20 border border-border/20">
              <p className="text-[9px] font-black text-muted-foreground uppercase tracking-widest mb-1">Unrealized</p>
              <p className={cn("text-xs font-black font-mono", getToneClass(risk_budget.unrealized_pnl >= 0 ? "positive" : "negative"))}>
                {fmtPnl(risk_budget.unrealized_pnl)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* ── Bottom Row: Top Positions + Predictive ── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">

        {/* Top Positions */}
        <div className="glass-card rounded-2xl p-6 overflow-hidden">
          <div className="flex items-center gap-2 mb-6 border-b border-border/40 pb-4">
             <TrendingUp className="h-4 w-4 text-primary" />
             <h3 className="text-xs font-black text-muted-foreground uppercase tracking-[0.2em]">Alpha Capture: Top P&L</h3>
          </div>
          
          {top_positions.length === 0 ? (
            <div className="py-12 text-center text-xs font-bold text-muted-foreground opacity-30 italic">
              NO ACTIVE EXPOSURE VECTORS
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="text-[10px] font-black text-muted-foreground border-b border-border/20 uppercase tracking-tighter">
                    <th className="pb-3 px-2">Symbol</th>
                    <th className="pb-3 px-2 text-right">P&L Abs</th>
                    <th className="pb-3 px-2 text-right">P&L %</th>
                    <th className="pb-3 px-2 text-right">Alpha Signal</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/10">
                  {top_positions.map((p) => (
                    <tr key={p.symbol} className="hover:bg-primary/[0.03] transition-colors group">
                      <td className="py-3 px-2">
                        <Badge variant="outline" className="font-mono text-[10px] font-black bg-background/40 group-hover:border-primary/50">
                          {p.symbol.replace("CRYPTO:", "")}
                        </Badge>
                      </td>
                      <td className={cn("py-3 px-2 text-right font-mono text-[11px] font-bold", getToneClass(p.pnl >= 0 ? "positive" : "negative"))}>
                        {fmtPnl(p.pnl)}
                      </td>
                      <td className={cn("py-3 px-2 text-right font-mono text-[11px] font-black", getToneClass(p.pnl_pct >= 0 ? "positive" : "negative"))}>
                        {p.pnl_pct >= 0 ? "+" : ""}{p.pnl_pct.toFixed(2)}%
                      </td>
                      <td className="py-3 px-2 text-right">
                        <Badge 
                          variant={p.signal_direction === "bullish" ? "positive" : p.signal_direction === "bearish" ? "negative" : "secondary"} 
                          className="text-[9px] h-4.5 font-bold uppercase"
                        >
                          {p.signal_direction ?? "NEUTRAL"}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Predictive Indicators */}
        <div className="glass-card rounded-2xl p-6 flex flex-col justify-between">
          <div className="flex items-center gap-2 mb-6 border-b border-border/40 pb-4">
             <BarChart3 className="h-4 w-4 text-primary" />
             <h3 className="text-xs font-black text-muted-foreground uppercase tracking-[0.2em]">Neural Predictive Analytics</h3>
          </div>

          <div className="space-y-6">
            <div className="p-4 rounded-2xl bg-background/30 border border-border/20">
              <div className="flex justify-between items-center mb-3">
                <span className="text-[11px] font-black text-muted-foreground uppercase tracking-widest">Regime Transition Entropy</span>
                {predictive.transition_direction && predictive.transition_direction !== "unknown" && (
                  <Badge variant="warning" className="text-[9px] h-4.5 animate-pulse uppercase">
                    DIR: {predictive.transition_direction}
                  </Badge>
                )}
              </div>
              <ProbGauge prob={predictive.transition_probability} />
            </div>

            {predictive.transition_size_mult !== null && predictive.transition_size_mult < 1 && (
              <div className="rounded-xl bg-warning/10 border border-warning/30 p-4 flex gap-3 animate-in slide-in-from-right-4">
                <TrendingDown className="h-4 w-4 text-warning shrink-0" />
                <p className="text-[11px] font-bold text-warning leading-relaxed uppercase tracking-tight">
                  Sleeve sizing dampened to {((predictive.transition_size_mult ?? 1) * 100).toFixed(0)}% due to high regime volatility.
                </p>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-2xl bg-background/20 border border-border/10">
                <p className="text-[9px] font-black text-muted-foreground uppercase mb-1 tracking-widest">RL Engine Updates</p>
                <p className="text-sm font-black font-mono text-foreground">{predictive.rl_total_updates?.toLocaleString() ?? "—"}</p>
              </div>
              <div className="p-4 rounded-2xl bg-background/20 border border-border/10">
                <p className="text-[9px] font-black text-muted-foreground uppercase mb-1 tracking-widest">Exploration (Epsilon)</p>
                <p className="text-sm font-black font-mono text-foreground">
                  {predictive.rl_epsilon !== null ? `${(predictive.rl_epsilon * 100).toFixed(1)}%` : "—"}
                </p>
              </div>
              <div className="p-4 rounded-2xl bg-background/20 border border-border/10">
                <p className="text-[9px] font-black text-muted-foreground uppercase mb-1 tracking-widest">Bayesian Vol Prob</p>
                <p className="text-sm font-black font-mono text-foreground">
                  {predictive.bayesian_vol_prob !== null ? `${((predictive.bayesian_vol_prob ?? 0) * 100).toFixed(1)}%` : "—"}
                </p>
              </div>
              <div className="p-4 rounded-2xl bg-background/20 border border-border/10">
                <p className="text-[9px] font-black text-muted-foreground uppercase mb-1 tracking-widest">Universe Scored</p>
                <p className="text-sm font-black font-mono text-foreground">
                  {predictive.universe_scored !== null ? `${predictive.universe_scored} syms` : "—"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
