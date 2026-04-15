"use client";

import { useEffect, useState } from "react";
import { Activity, Shield, Zap, TrendingUp, TrendingDown, RefreshCw, BarChart3, AlertCircle, Clock, LayoutGrid, Search, Filter } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, getToneClass } from "@/lib/utils";

// ── Types ──────────────────────────────────────────────────────────────────────

interface GateInfo {
  fires: number;
  blocks: number;
  block_rate: number;
  total_decisions: number;
}

interface BlockedAnalysis {
  total_decisions: number;
  total_blocked: number;
  block_rate: number;
  by_first_gate: Record<string, number>;
}

interface BlockedSymbol {
  symbol: string;
  block_rate: number;
  blocked: number;
  total: number;
}

interface DiagnosticsReport {
  lookback_days: number;
  total_records: number;
  entered: number;
  completed_trades: number;
  overall_win_rate: number | null;
  gate_attribution: Record<string, GateInfo>;
  blocked_analysis: BlockedAnalysis;
  most_blocked_symbols: BlockedSymbol[];
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function pct(v: number | null | undefined): string {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function gateVariant(blockRate: number): "positive" | "warning" | "negative" {
  if (blockRate >= 0.6) return "negative";
  if (blockRate >= 0.3) return "warning";
  return "positive";
}

// ── Component ────────────────────────────────────────────────────────────────

export default function RegimeSharpePanel() {
  const [data, setData] = useState<DiagnosticsReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lookback, setLookback] = useState(7);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/v1/regime-sharpe?lookback_days=${lookback}`,
        { cache: "no-store" }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load diagnostics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 60_000);
    return () => clearInterval(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lookback]);

  if (loading && !data) {
    return (
      <div className="glass-card rounded-2xl p-12 flex flex-col items-center justify-center gap-4 text-muted-foreground animate-pulse">
        <RefreshCw className="h-8 w-8 animate-spin opacity-20" />
        <p className="text-xs font-bold uppercase tracking-widest">Compiling Gate Diagnostics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-2xl border border-negative/30 bg-negative/10 p-6 flex flex-col items-center gap-4 animate-in shake duration-300">
        <div className="flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-negative" />
          <span className="text-sm font-bold text-negative">Diagnostic Capture Failure: {error}</span>
        </div>
        <Button variant="outline" size="sm" onClick={load} className="text-xs font-black uppercase">
          Re-Initialize Probe
        </Button>
      </div>
    );
  }

  if (!data) return null;

  const gates = Object.entries(data.gate_attribution ?? {});
  const topGates = gates.slice(0, 10);

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/40 pb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-primary/10 text-primary shadow-inner">
             <Shield size={20} className="text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-black text-foreground uppercase tracking-tight">Gate Diagnostics Leaderboard</h2>
            <p className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground font-black">
              Execution Barrier Analysis
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-background/40 border border-border/20 rounded-xl px-3 py-1.5">
             <Clock size={12} className="text-muted-foreground" />
             <select
               value={lookback}
               onChange={(e) => setLookback(Number(e.target.value))}
               className="bg-transparent text-[11px] font-black uppercase focus:outline-none cursor-pointer"
             >
               {[1, 3, 7, 14, 30].map((d) => (
                 <option key={d} value={d} className="bg-background text-foreground">
                   {d} Day Roll
                 </option>
               ))}
             </select>
          </div>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={load} 
            disabled={loading}
            className="text-[10px] font-black uppercase tracking-widest hover:bg-primary/10"
          >
            {loading ? <RefreshCw className="h-3 w-3 animate-spin mr-2" /> : <RefreshCw className="h-3 w-3 mr-2" />}
            Refresh
          </Button>
        </div>
      </div>

      {/* Summary row */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {[
          {
            label: "Total Decisions",
            value: data.total_records.toLocaleString(),
            icon: Activity
          },
          { 
            label: "Actual Entries", 
            value: data.entered.toLocaleString(),
            icon: TrendingUp
          },
          {
            label: "Net Block Rate",
            value: pct(data.blocked_analysis.block_rate),
            tone: gateVariant(data.blocked_analysis.block_rate),
            icon: Shield
          },
          {
            label: "System Win Rate",
            value: data.overall_win_rate != null ? pct(data.overall_win_rate) : "—",
            tone: (data.overall_win_rate ?? 0) >= 0.5 ? "positive" : "warning",
            icon: Zap
          },
        ].map(({ label, value, tone, icon: Icon }) => (
          <div
            key={label}
            className="glass-card rounded-2xl p-5 flex flex-col items-center text-center group transition-all hover:bg-primary/[0.02]"
          >
            <div className="p-2 rounded-lg bg-background/40 mb-3 group-hover:scale-110 transition-transform">
               <Icon size={14} className="text-muted-foreground" />
            </div>
            <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest mb-1">{label}</p>
            <p className={cn("text-xl font-black font-mono tracking-tighter", tone ? getToneClass(tone as any) : "text-foreground")}>{value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* Gate attribution table */}
        {topGates.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 px-1">
               <Filter size={14} className="text-primary" />
               <h3 className="text-[11px] font-black text-muted-foreground uppercase tracking-widest">
                 Critical Blocking Gates (Last {lookback}D)
               </h3>
            </div>
            <div className="glass-card rounded-2xl border border-border/40 overflow-hidden shadow-2xl shadow-black/20">
              <div className="overflow-x-auto custom-scrollbar">
                <table className="w-full text-left">
                  <thead>
                    <tr className="text-[10px] font-black text-muted-foreground bg-background/60 border-b border-border/20 uppercase tracking-tighter">
                      <th className="py-4 px-4">Gate Identifier</th>
                      <th className="py-4 px-4 text-right">Blocks</th>
                      <th className="py-4 px-4 text-right">Fires</th>
                      <th className="py-4 px-4 text-right pr-6">Block Rate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/10">
                    {topGates.map(([gate, info]) => (
                      <tr
                        key={gate}
                        className="hover:bg-primary/[0.03] transition-colors group"
                      >
                        <td className="py-4 px-4">
                          <code className="text-[10px] font-mono font-black text-foreground group-hover:text-primary transition-colors">{gate}</code>
                        </td>
                        <td className="py-4 px-4 text-right font-mono text-[11px] font-bold text-foreground">{info.blocks.toLocaleString()}</td>
                        <td className="py-4 px-4 text-right font-mono text-[11px] text-muted-foreground">{info.fires.toLocaleString()}</td>
                        <td className="py-4 px-4 text-right pr-6">
                           <Badge variant={gateVariant(info.block_rate)} className="font-mono text-[10px] min-w-[5ch] justify-center">
                             {pct(info.block_rate)}
                           </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Most blocked symbols */}
        {data.most_blocked_symbols.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 px-1">
               <BarChart3 size={14} className="text-primary" />
               <h3 className="text-[11px] font-black text-muted-foreground uppercase tracking-widest">
                 Systemic Symbol Rejections
               </h3>
            </div>
            <div className="glass-card rounded-2xl border border-border/40 overflow-hidden shadow-2xl shadow-black/20">
              <div className="overflow-x-auto custom-scrollbar">
                <table className="w-full text-left">
                  <thead>
                    <tr className="text-[10px] font-black text-muted-foreground bg-background/60 border-b border-border/20 uppercase tracking-tighter">
                      <th className="py-4 px-4">Symbol</th>
                      <th className="py-4 px-4 text-right">Rejected</th>
                      <th className="py-4 px-4 text-right">Evaluated</th>
                      <th className="py-4 px-4 text-right pr-6">Rejection %</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/10">
                    {data.most_blocked_symbols.map((row) => (
                      <tr
                        key={row.symbol}
                        className="hover:bg-primary/[0.03] transition-colors group"
                      >
                        <td className="py-4 px-4">
                          <Badge variant="outline" className="font-mono text-[10px] font-black bg-background/40 group-hover:border-primary/50">
                            {row.symbol}
                          </Badge>
                        </td>
                        <td className="py-4 px-4 text-right font-mono text-[11px] font-bold text-foreground">{row.blocked.toLocaleString()}</td>
                        <td className="py-4 px-4 text-right font-mono text-[11px] text-muted-foreground">{row.total.toLocaleString()}</td>
                        <td className="py-4 px-4 text-right pr-6">
                           <Badge variant={gateVariant(row.block_rate)} className="font-mono text-[10px] min-w-[5ch] justify-center">
                             {pct(row.block_rate)}
                           </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* First-gate breakdown */}
      {Object.keys(data.blocked_analysis.by_first_gate ?? {}).length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 px-1">
             <LayoutGrid size={14} className="text-primary" />
             <h3 className="text-[11px] font-black text-muted-foreground uppercase tracking-widest">
               Primary Rejection Attribution
             </h3>
          </div>
          <div className="glass-card rounded-2xl p-6 grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-4">
            {Object.entries(data.blocked_analysis.by_first_gate)
              .slice(0, 10)
              .map(([gate, count]) => {
                const ratio =
                  data.blocked_analysis.total_blocked > 0
                    ? count / data.blocked_analysis.total_blocked
                    : 0;
                return (
                  <div key={gate} className="space-y-1.5 group">
                    <div className="flex items-center justify-between text-[11px] font-bold">
                       <code className="text-[10px] font-mono font-black text-muted-foreground truncate max-w-[200px] group-hover:text-foreground transition-colors">{gate}</code>
                       <span className="text-foreground">{count.toLocaleString()} <span className="text-muted-foreground ml-1">({pct(ratio)})</span></span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-background/40 border border-border/10 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-primary/60 group-hover:bg-primary transition-all duration-700"
                        style={{ width: `${(ratio * 100).toFixed(1)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {data.total_records === 0 && (
        <div className="glass-card rounded-2xl border-dashed border-2 py-20 flex flex-col items-center gap-4 text-muted-foreground opacity-40">
           <Search size={32} strokeWidth={1} />
           <p className="text-sm font-bold uppercase tracking-widest text-center">
             Zero Decision Records in {lookback}D Window
             <br />
             <span className="text-[10px] font-medium normal-case block mt-2">Diagnostics will hydrate once the strategy enters active evaluation.</span>
           </p>
        </div>
      )}
    </div>
  );
}
