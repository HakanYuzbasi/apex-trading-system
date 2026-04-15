"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";
import { Activity, Shield, Zap, TrendingUp, TrendingDown, RefreshCw, BarChart3, AlertCircle, Clock, LineChart } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn, getToneClass } from "@/lib/utils";

interface WinRates {
  paper: number;
  live: number;
  n: number;
}

interface PaperTrade {
  symbol: string;
  side: string;
  entry_price: number;
  exit_price: number;
  notional: number;
  pnl_usd: number;
  live_pnl_usd: number;
  shortfall_usd: number;
  entry_ts: number;
  exit_ts: number;
}

interface PaperAccountData {
  available: boolean;
  note?: string;
  open_positions?: number;
  closed_trades?: number;
  paper_total_pnl?: number;
  live_total_pnl?: number;
  implementation_shortfall_usd?: number;
  shortfall_pct?: number;
  avg_shortfall_per_trade?: number;
  win_rates?: WinRates;
  day_start_ts?: number;
  recent_trades?: PaperTrade[];
}

function fmtUsd(v?: number): string {
  if (v == null) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtTs(epoch?: number): string {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toISOString().slice(11, 16) + " UTC";
}

function ShortfallGauge({ pct }: { pct: number }) {
  const abs = Math.min(Math.abs(pct), 100);
  const tone = abs < 5 ? "positive" : abs < 15 ? "warning" : "negative";
  const barColor = tone === "positive" ? "bg-positive" : tone === "warning" ? "bg-warning" : "bg-negative";
  
  return (
    <div className="space-y-2">
      <div className="h-2 w-full rounded-full bg-background/40 border border-border/20 overflow-hidden">
        <div className={cn("h-full rounded-full transition-all duration-700", barColor)} style={{ width: `${abs}%` }} />
      </div>
      <div className="flex justify-between items-center">
        <p className={cn("text-[10px] font-black uppercase tracking-[0.1em]", getToneClass(tone))}>
          {pct.toFixed(1)}% Consumption of Paper P&L
        </p>
        <Badge variant={tone} className="text-[9px] h-4 px-1.5 font-bold">{abs < 5 ? "LOW_SLIPPAGE" : abs < 15 ? "MODERATE_FRICTION" : "HIGH_IMPACT"}</Badge>
      </div>
    </div>
  );
}

export default function PaperAccountPanel() {
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<PaperAccountData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch("/api/v1/paper-account", {
          headers: token ? { authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    const id = setInterval(load, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [token]);

  if (loading)
    return (
      <div className="glass-card rounded-2xl p-12 flex flex-col items-center justify-center gap-4 text-muted-foreground animate-pulse">
        <RefreshCw className="h-8 w-8 animate-spin opacity-20" />
        <p className="text-xs font-bold uppercase tracking-widest">Hydrating Shadow States...</p>
      </div>
    );

  if (error)
    return (
      <div className="rounded-2xl border border-negative/30 bg-negative/10 p-6 flex items-center gap-3 animate-in shake duration-300">
        <AlertCircle className="h-5 w-5 text-negative" />
        <span className="text-sm font-bold text-negative">Account Engine Error: {error}</span>
      </div>
    );

  if (!data || !data.available) {
    return (
      <div className="glass-card rounded-2xl p-12 flex flex-col items-center justify-center gap-4 text-muted-foreground opacity-50">
        <Shield className="h-8 w-8 opacity-20" />
        <p className="text-xs font-bold uppercase tracking-widest text-center">
          {data?.note ?? "Shadow Implementation Unavailable — Simulation Offline"}
        </p>
      </div>
    );
  }

  const shortfall = data.implementation_shortfall_usd ?? 0;
  const shortfallPct = data.shortfall_pct ?? 0;
  const wr = data.win_rates ?? { paper: 0, live: 0, n: 0 };
  const trades = data.recent_trades ?? [];

  return (
    <div className="p-6 space-y-8 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/40 pb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-primary/10 text-primary shadow-inner">
             <LineChart size={20} />
          </div>
          <div>
            <h2 className="text-lg font-black text-foreground uppercase tracking-tight">Shadow Paper Account</h2>
            <div className="flex items-center gap-2 mt-0.5">
               <Badge variant="outline" className="text-[9px] h-4.5 px-1.5 font-bold uppercase bg-background/40">Slippage Tracker</Badge>
               <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">
                 {data.closed_trades ?? 0} Execution Samples Found
               </span>
            </div>
          </div>
        </div>
        <div className="text-right">
           <p className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em] mb-1">Epoch Start</p>
           <Badge variant="secondary" className="font-mono text-[10px] h-6 px-3 bg-background/50">
             {fmtTs(data.day_start_ts)}
           </Badge>
        </div>
      </div>

      {/* P&L comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card rounded-2xl p-5 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
             <Activity size={48} className="text-primary" />
          </div>
          <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest mb-3">Theoretical P&L (Mid-Price)</p>
          <p className={cn("text-3xl font-black font-mono tracking-tighter", getToneClass((data.paper_total_pnl ?? 0) >= 0 ? "positive" : "negative"))}>
            {fmtUsd(data.paper_total_pnl)}
          </p>
          <p className="text-[10px] font-bold text-muted-foreground/60 uppercase mt-3 tracking-tight">Zero-impact simulation model</p>
        </div>
        
        <div className="glass-card rounded-2xl p-5 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
             <Zap size={48} className="text-primary" />
          </div>
          <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest mb-3">Realized P&L (Frictional)</p>
          <p className={cn("text-3xl font-black font-mono tracking-tighter", getToneClass((data.live_total_pnl ?? 0) >= 0 ? "positive" : "negative"))}>
            {fmtUsd(data.live_total_pnl)}
          </p>
          <p className="text-[10px] font-bold text-muted-foreground/60 uppercase mt-3 tracking-tight">Post-Commission + Slippage</p>
        </div>
      </div>

      {/* Implementation shortfall */}
      <div className="glass-card rounded-2xl p-6 space-y-6 bg-gradient-to-br from-background/40 to-muted/20">
        <div className="flex flex-wrap items-center justify-between gap-6">
          <div className="space-y-1">
            <p className="text-[11px] font-black text-muted-foreground uppercase tracking-[0.2em]">Total Implementation Shortfall</p>
            <p className={cn("text-4xl font-black font-mono tracking-tighter", shortfall <= 0 ? "text-positive" : shortfall < 50 ? "text-warning" : "text-negative")}>
              {fmtUsd(shortfall)}
            </p>
          </div>
          <div className="p-4 rounded-2xl bg-background/30 border border-border/20 text-right min-w-[160px]">
            <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest mb-1">Avg Loss / Trade</p>
            <p className="text-xl font-black font-mono text-foreground tracking-tight">
              {fmtUsd(data.avg_shortfall_per_trade)}
            </p>
          </div>
        </div>
        <ShortfallGauge pct={shortfallPct} />
      </div>

      {/* Win rate comparison */}
      {wr.n > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 px-1">
             <BarChart3 size={14} className="text-primary" />
             <h3 className="text-[11px] font-black text-muted-foreground uppercase tracking-widest">
               Alpha Retention Analysis ({wr.n} Matched Execution Pairs)
             </h3>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="glass-card rounded-2xl p-4 text-center border-l-4 border-l-primary/40">
              <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest mb-1">Simulated WR</p>
              <p className="text-2xl font-black font-mono text-primary tracking-tighter">{(wr.paper * 100).toFixed(0)}%</p>
            </div>
            <div className="glass-card rounded-2xl p-4 text-center border-l-4 border-l-positive/40">
              <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest mb-1">Observed WR</p>
              <p className={cn("text-2xl font-black font-mono tracking-tighter", wr.live >= wr.paper - 0.05 ? "text-positive" : "text-warning")}>
                {(wr.live * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Recent trades */}
      {trades.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 px-1">
             <Clock size={14} className="text-primary" />
             <h3 className="text-[11px] font-black text-muted-foreground uppercase tracking-widest">Execution Audit Log</h3>
          </div>
          
          <div className="glass-card rounded-2xl border border-border/40 overflow-hidden shadow-2xl shadow-black/20">
            <div className="overflow-x-auto custom-scrollbar">
              <table className="w-full text-left">
                <thead>
                  <tr className="text-[10px] font-black text-muted-foreground bg-background/60 border-b border-border/20 uppercase tracking-tighter">
                    <th className="py-4 px-4">Symbol</th>
                    <th className="py-4 px-4 text-right">Theory P&L</th>
                    <th className="py-4 px-4 text-right">Actual P&L</th>
                    <th className="py-4 px-4 text-right">Gap (Slippage)</th>
                    <th className="py-4 px-4 text-right pr-6">Exit TS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/10">
                  {trades.slice(0, 10).map((t, i) => {
                    const gap = t.shortfall_usd;
                    return (
                      <tr key={i} className="hover:bg-primary/[0.03] transition-colors group">
                        <td className="py-4 px-4">
                          <Badge variant="outline" className="font-mono text-[10px] font-black bg-background/40 group-hover:border-primary/40">
                            {t.symbol.replace("CRYPTO:", "").replace("/USD", "")}
                          </Badge>
                        </td>
                        <td className={cn("py-4 px-4 text-right font-mono text-[11px] font-bold", getToneClass(t.pnl_usd >= 0 ? "positive" : "negative"))}>
                          {fmtUsd(t.pnl_usd)}
                        </td>
                        <td className={cn("py-4 px-4 text-right font-mono text-[11px] font-bold", getToneClass(t.live_pnl_usd >= 0 ? "positive" : "negative"))}>
                          {t.live_pnl_usd === 0 ? "—" : fmtUsd(t.live_pnl_usd)}
                        </td>
                        <td className={cn("py-4 px-4 text-right font-mono text-[11px] font-black", shortfall <= 0 ? "text-positive" : shortfall < 50 ? "text-warning" : "text-negative")}>
                          {t.live_pnl_usd === 0 ? "—" : fmtUsd(gap)}
                        </td>
                        <td className="py-4 px-4 text-right font-mono text-[10px] font-bold text-muted-foreground pr-6 italic uppercase">
                          {fmtTs(t.exit_ts)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {trades.length === 0 && (
        <div className="glass-card rounded-2xl border-dashed border-2 py-16 flex flex-col items-center gap-4 text-muted-foreground opacity-40">
           <Activity className="h-8 w-8" />
           <p className="text-xs font-bold uppercase tracking-widest">Simulation Log is Empty — Awaiting Initial Executions</p>
        </div>
      )}
    </div>
  );
}
