"use client";

import { useEffect, useState } from "react";
import { Activity, Shield, Zap, TrendingUp, TrendingDown, RefreshCw, BarChart3, AlertCircle, Clock, LayoutGrid, Target, Microscope } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, getToneClass } from "@/lib/utils";

// ── Types ──────────────────────────────────────────────────────────────────────

interface ICTrackerData {
  summary: Record<string, number>;
  dead_features: string[];
  strong_features: string[];
  pending_count: number;
  observation_counts: Record<string, number>;
  error?: string;
}

interface BucketStats {
  label: string;
  n_obs: number;
  ic: number;
  hit_rate: number;
  mean_return: number;
}

interface RegimeDecay {
  optimal_hold_hours: number;
  alpha_half_life: number | null;
  peak_ic: number;
  buckets: BucketStats[];
}

interface AlphaDecayData {
  total_trades: number;
  updated_at: string;
  regimes: Record<string, RegimeDecay>;
  error?: string;
}

interface ICReport {
  ic_tracker?: ICTrackerData;
  alpha_decay?: AlphaDecayData;
  error?: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function icTone(ic: number): "positive" | "warning" | "negative" | "neutral" {
  if (ic >= 0.05) return "positive";
  if (ic >= 0.03) return "warning";
  if (ic >= 0.015) return "neutral";
  return "negative";
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function ICReportPanel() {
  const [data, setData] = useState<ICReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/ic-report", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 60_000); // refresh every 60s
    return () => clearInterval(id);
  }, []);

  if (loading && !data)
    return (
      <div className="glass-card rounded-2xl p-12 flex flex-col items-center justify-center gap-4 text-muted-foreground animate-pulse">
        <RefreshCw className="h-8 w-8 animate-spin opacity-20" />
        <p className="text-xs font-bold uppercase tracking-widest">Running Information Analysis...</p>
      </div>
    );

  if (error)
    return (
      <div className="rounded-2xl border border-negative/30 bg-negative/10 p-6 flex flex-col items-center gap-4 animate-in shake duration-300">
        <div className="flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-negative" />
          <span className="text-sm font-bold text-negative">IC Pipeline Failure: {error}</span>
        </div>
        <Button variant="outline" size="sm" onClick={load} className="text-xs font-black uppercase">
          Retry Analysis Probe
        </Button>
      </div>
    );

  const ict = data?.ic_tracker;
  const adc = data?.alpha_decay;

  return (
    <div className="space-y-8 animate-in fade-in duration-500 p-2">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/40 pb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-primary/10 text-primary shadow-inner">
             <Target size={20} className="text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-black text-foreground uppercase tracking-tight">Information Coefficient (IC) Report</h2>
            <div className="flex items-center gap-2 mt-0.5">
               <Badge variant="outline" className="text-[9px] h-4.5 px-1.5 font-bold uppercase bg-background/40">Spearman Rank</Badge>
               <p className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground font-black">
                 Signal Decay & Component Attribution
               </p>
            </div>
          </div>
        </div>
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={load} 
          disabled={loading}
          className="text-[10px] font-black uppercase tracking-widest hover:bg-primary/10"
        >
          {loading ? <RefreshCw className="h-3 w-3 animate-spin mr-2" /> : <RefreshCw className="h-3 w-3 mr-2" />}
          Re-Analyze
        </Button>
      </div>

      {/* IC Tracker Section */}
      {ict && !ict.error ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
               <BarChart3 size={14} className="text-primary" />
               <h3 className="text-sm font-black text-muted-foreground uppercase tracking-widest">
                 Feature Predictive Power
               </h3>
            </div>
            <Badge variant="outline" className="text-[10px] font-bold bg-background/40">
              PENDING_FILLS: {ict.pending_count}
            </Badge>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Dead / Strong callouts */}
            {ict.dead_features.length > 0 && (
              <div className="rounded-2xl border border-negative/20 bg-negative/5 p-4 flex items-start gap-3 animate-in slide-in-from-left-4">
                <AlertCircle className="h-4 w-4 text-negative shrink-0 mt-0.5" />
                <div>
                  <p className="text-[11px] font-black text-negative uppercase tracking-tight mb-1">Dampened Features (IC &lt; 0.015)</p>
                  <p className="text-[10px] font-bold text-negative/70 leading-relaxed uppercase">
                    {ict.dead_features.join(", ")} · Signal contribution reduced by 20%
                  </p>
                </div>
              </div>
            )}
            {ict.strong_features.length > 0 && (
              <div className="rounded-2xl border border-positive/20 bg-positive/5 p-4 flex items-start gap-3 animate-in slide-in-from-right-4">
                <Zap className="h-4 w-4 text-positive shrink-0 mt-0.5" />
                <div>
                  <p className="text-[11px] font-black text-positive uppercase tracking-tight mb-1">Alpha Boosted (IC &gt; 0.05)</p>
                  <p className="text-[10px] font-bold text-positive/70 leading-relaxed uppercase">
                    {ict.strong_features.join(", ")} · Model confidence amplified by 8%
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Feature table */}
          <div className="glass-card rounded-2xl border border-border/40 overflow-hidden shadow-2xl shadow-black/20">
            <div className="overflow-x-auto custom-scrollbar">
              <table className="w-full text-left">
                <thead>
                  <tr className="text-[10px] font-black text-muted-foreground bg-background/60 border-b border-border/20 uppercase tracking-tighter">
                    <th className="py-4 px-4">Model Component</th>
                    <th className="py-4 px-4 text-right">30D Rolling IC</th>
                    <th className="py-4 px-4 text-right">Samples</th>
                    <th className="py-4 px-4 text-center pr-6">Attribution State</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/10">
                  {Object.entries(ict.summary).map(([feat, ic]) => (
                    <tr
                      key={feat}
                      className="hover:bg-primary/[0.03] transition-colors group"
                    >
                      <td className="py-4 px-4">
                        <code className="text-[11px] font-mono font-black text-foreground group-hover:text-primary transition-colors">{feat}</code>
                      </td>
                      <td className={cn("py-4 px-4 text-right font-mono text-[12px] font-black", getToneClass(icTone(ic)))}>
                        {ic.toFixed(4)}
                      </td>
                      <td className="py-4 px-4 text-right font-mono text-[11px] font-bold text-muted-foreground">
                        {ict.observation_counts[feat] ?? "—"}
                      </td>
                      <td className="py-4 px-4 text-center pr-6">
                        {ict.strong_features.includes(feat) ? (
                          <Badge variant="positive" className="text-[9px] h-4.5 font-black uppercase tracking-widest bg-positive/10 ring-1 ring-positive/30">Alpha_Lead</Badge>
                        ) : ict.dead_features.includes(feat) ? (
                          <Badge variant="negative" className="text-[9px] h-4.5 font-black uppercase tracking-widest bg-negative/10 ring-1 ring-negative/30">Dampened</Badge>
                        ) : (
                          <Badge variant="secondary" className="text-[9px] h-4.5 font-black uppercase tracking-widest">Stable</Badge>
                        )}
                      </td>
                    </tr>
                  ))}
                  {Object.keys(ict.summary).length === 0 && (
                    <tr>
                      <td colSpan={4} className="py-12 text-center text-[11px] font-black text-muted-foreground uppercase tracking-widest opacity-30 italic">
                        Insufficient observation density for feature attribution
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : (
        ict?.error && (
          <div className="p-4 rounded-xl bg-negative/10 border border-negative/30 text-[10px] font-black text-negative uppercase">
            IC Tracker Synchronization Error: {ict.error}
          </div>
        )
      )}

      {/* Alpha Decay Section */}
      {adc && !adc.error ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
               <Clock size={14} className="text-primary" />
               <h3 className="text-sm font-black text-muted-foreground uppercase tracking-widest">
                 Horizon Alpha Decay
               </h3>
            </div>
            <Badge variant="outline" className="text-[10px] font-bold bg-background/40">
              SAMPLE_SIZE: {adc.total_trades} TRADES
            </Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(adc.regimes).map(([regime, rd]) => (
              <div
                key={regime}
                className="glass-card rounded-2xl p-5 space-y-5 border-l-4 border-l-primary/30"
              >
                <div className="flex items-center justify-between border-b border-border/20 pb-3">
                  <h4 className="text-xs font-black uppercase text-foreground tracking-widest">
                    Regime: {regime}
                  </h4>
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className="text-[9px] font-bold px-1.5 h-4.5 bg-background/40">
                      OPT: {rd.optimal_hold_hours}H
                    </Badge>
                    {rd.alpha_half_life != null && (
                      <Badge variant="warning" className="text-[9px] font-bold px-1.5 h-4.5">
                        T 1/2: {rd.alpha_half_life}H
                      </Badge>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-4 gap-3">
                  {rd.buckets.map((b) => (
                    <div key={b.label} className="p-3 rounded-xl bg-background/40 border border-border/10 flex flex-col items-center">
                      <div
                        className={cn("text-[11px] font-black font-mono mb-1.5 tracking-tighter", getToneClass(icTone(b.n_obs >= 10 ? b.ic : 0)))}
                      >
                        {b.n_obs >= 10 ? b.ic.toFixed(3) : "—"}
                      </div>
                      <div className="text-[9px] font-black text-muted-foreground uppercase tracking-tighter mb-1">{b.label}</div>
                      <Badge variant="outline" className="text-[8px] h-3.5 px-1 font-bold bg-background/20 opacity-40">N:{b.n_obs}</Badge>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          {Object.entries(adc.regimes).length === 0 && (
            <div className="glass-card rounded-2xl border-dashed border-2 py-12 flex flex-col items-center gap-4 text-muted-foreground opacity-40">
               <Activity className="h-8 w-8" />
               <p className="text-xs font-bold uppercase tracking-widest text-center">
                 Awaiting Cross-Regime Alpha Persistence Data
               </p>
            </div>
          )}
        </div>
      ) : (
        adc?.error && (
          <div className="p-4 rounded-xl bg-negative/10 border border-negative/30 text-[10px] font-black text-negative uppercase">
            Alpha Decay Matrix Error: {adc.error}
          </div>
        )
      )}
    </div>
  );
}
