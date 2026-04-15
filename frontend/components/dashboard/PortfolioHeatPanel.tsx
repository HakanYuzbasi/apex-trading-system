"use client";

import { useEffect, useState } from "react";
import { Activity, Shield, Zap, TrendingUp, TrendingDown, RefreshCw, BarChart3, AlertCircle, Clock, LayoutGrid, Flame } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn, getToneClass } from "@/lib/utils";

interface PositionRow {
  symbol: string;
  qty: number;
  notional: number;
  asset_class: string;
  weight_pct: number;
}

interface AssetClassInfo {
  count: number;
  notional: number;
  weight_pct: number;
}

interface AlphaDecay {
  optimal_hold_hours: number | null;
  alpha_half_life: number | null;
}

interface ModelDrift {
  health: string;
  should_retrain: boolean;
  ic_current: number;
  hit_rate_current: number;
}

interface PortfolioHeat {
  positions: PositionRow[];
  by_asset_class: Record<string, AssetClassInfo>;
  total_notional: number;
  position_count: number;
  hhi_concentration: number;
  regime: string;
  vix: number;
  alpha_decay: AlphaDecay | null;
  model_drift: ModelDrift | null;
}

function heatTone(weightPct: number): "positive" | "warning" | "negative" | "secondary" {
  if (weightPct >= 20) return "negative";
  if (weightPct >= 12) return "warning";
  if (weightPct >= 6) return "secondary";
  return "positive";
}

function driftVariant(health: string): "positive" | "warning" | "negative" {
  if (health === "critical") return "negative";
  if (health === "degrading") return "warning";
  return "positive";
}

function hhiVariant(hhi: number): "positive" | "warning" | "negative" {
  if (hhi > 0.40) return "negative";
  if (hhi > 0.20) return "warning";
  return "positive";
}

export default function PortfolioHeatPanel() {
  const [heat, setHeat] = useState<PortfolioHeat | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch("/api/v1/portfolio-heat", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: PortfolioHeat = await res.json();
        if (!cancelled) setHeat(data);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    const interval = setInterval(load, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (loading)
    return (
      <div className="glass-card rounded-2xl p-12 flex flex-col items-center justify-center gap-4 text-muted-foreground animate-pulse">
        <RefreshCw className="h-8 w-8 animate-spin opacity-20" />
        <p className="text-xs font-bold uppercase tracking-widest">Scanning Portfolio Heat...</p>
      </div>
    );

  if (error || !heat)
    return (
      <div className="rounded-2xl border border-negative/30 bg-negative/10 p-6 flex items-center gap-3 animate-in shake duration-300">
        <AlertCircle className="h-5 w-5 text-negative" />
        <span className="text-sm font-bold text-negative">Telemetery Failure: {error ?? "Null data"}</span>
      </div>
    );

  return (
    <div className="glass-card rounded-2xl p-6 space-y-6 animate-in fade-in duration-500 h-full flex flex-col">
      {/* Header row */}
      <div className="flex items-center justify-between border-b border-border/40 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-primary/10 text-primary shadow-inner">
             <Flame size={20} className="text-primary animate-pulse" />
          </div>
          <div>
            <h3 className="text-sm font-black text-foreground uppercase tracking-tight">Portfolio Heat</h3>
            <p className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground font-black">
              Concentration Metrics
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          <Badge variant="outline" className="text-[10px] font-bold bg-background/40">REGIME: {heat.regime}</Badge>
          <Badge variant="outline" className="text-[10px] font-bold bg-background/40">VIX: {heat.vix?.toFixed(1) ?? "–"}</Badge>
        </div>
      </div>

      {/* Summary row */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 rounded-2xl bg-background/20 border border-border/10 text-center">
          <p className="text-[9px] font-black text-muted-foreground uppercase tracking-widest mb-1">Pos Count</p>
          <p className="text-xl font-black font-mono text-foreground tracking-tighter">{heat.position_count}</p>
        </div>
        <div className="p-4 rounded-2xl bg-background/20 border border-border/10 text-center">
          <p className="text-[9px] font-black text-muted-foreground uppercase tracking-widest mb-1">Total Exp</p>
          <p className="text-xl font-black font-mono text-foreground tracking-tighter">
            ${(heat.total_notional / 1000).toFixed(1)}k
          </p>
        </div>
        <div className="p-4 rounded-2xl bg-background/20 border border-border/10 text-center relative overflow-hidden group">
          <p className="text-[9px] font-black text-muted-foreground uppercase tracking-widest mb-1">HHI Index</p>
          <p className={cn("text-xl font-black font-mono tracking-tighter relative z-10", getToneClass(hhiVariant(heat.hhi_concentration)))}>
            {heat.hhi_concentration.toFixed(2)}
          </p>
          <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
      </div>

      {/* Content wrapper for columns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
        
        {/* Left Column: Asset Class + Model Drift */}
        <div className="space-y-6 flex flex-col">
          {/* Asset class breakdown */}
          <div className="space-y-4">
            <h4 className="text-[11px] font-black text-muted-foreground uppercase tracking-widest px-1">Asset Allocation</h4>
            <div className="space-y-3">
              {Object.entries(heat.by_asset_class).map(([ac, info]) => (
                <div key={ac} className="space-y-1.5">
                  <div className="flex items-center justify-between text-[11px] font-bold">
                    <span className="uppercase tracking-tight text-foreground/80">{ac}</span>
                    <span className="font-mono text-muted-foreground">${(info.notional / 1000).toFixed(1)}k <span className="text-foreground ml-1">({info.weight_pct.toFixed(0)}%)</span></span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-background/40 border border-border/10 overflow-hidden">
                    <div
                      className={cn("h-full rounded-full transition-all duration-700", ac === "crypto" ? "bg-primary" : "bg-emerald-500")}
                      style={{ width: `${info.weight_pct}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-auto space-y-4">
            {/* Alpha decay hint */}
            {heat.alpha_decay && (
              <div className="p-4 rounded-2xl bg-background/40 border border-border/20 flex flex-wrap gap-x-6 gap-y-2">
                <div className="flex items-center gap-2">
                   <Clock size={12} className="text-muted-foreground" />
                   <span className="text-[10px] font-bold text-muted-foreground uppercase">Optimal Hold: <span className="text-foreground">{heat.alpha_decay.optimal_hold_hours?.toFixed(1) ?? "–"}H</span></span>
                </div>
                {heat.alpha_decay.alpha_half_life != null && (
                  <div className="flex items-center gap-2">
                     <Activity size={12} className="text-muted-foreground" />
                     <span className="text-[10px] font-bold text-muted-foreground uppercase">α Half-Life: <span className="text-foreground">{heat.alpha_decay.alpha_half_life.toFixed(1)}H</span></span>
                  </div>
                )}
              </div>
            )}

            {/* Model drift badge */}
            {heat.model_drift && (
              <div className="p-4 rounded-2xl bg-background/40 border border-border/20 space-y-3">
                <div className="flex items-center justify-between">
                  <Badge variant={driftVariant(heat.model_drift.health)} className="text-[10px] font-black h-5 px-2">
                    MODEL: {heat.model_drift.health.toUpperCase()}
                  </Badge>
                  {heat.model_drift.should_retrain && (
                    <Badge variant="warning" className="text-[10px] font-black h-5 px-2 animate-pulse">
                      RE-TRAIN REQUIRED
                    </Badge>
                  )}
                </div>
                <div className="flex items-center gap-4 text-[11px] font-mono font-bold text-muted-foreground">
                  <span>IC: <span className="text-foreground">{heat.model_drift.ic_current.toFixed(3)}</span></span>
                  <span className="text-border/40">|</span>
                  <span>HR: <span className={cn(heat.model_drift.hit_rate_current >= 0.5 ? "text-positive" : "text-warning")}>{(heat.model_drift.hit_rate_current * 100).toFixed(0)}%</span></span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Position Heatmap Table */}
        <div className="flex flex-col min-h-0">
          <h4 className="text-[11px] font-black text-muted-foreground uppercase tracking-widest px-1 mb-4">Weight Concentration</h4>
          <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar space-y-1.5">
            {[...heat.positions]
              .sort((a, b) => b.weight_pct - a.weight_pct)
              .map((pos) => {
                const tone = heatTone(pos.weight_pct);
                return (
                  <div
                    key={pos.symbol}
                    className="flex items-center justify-between p-3 rounded-xl bg-background/30 border border-border/10 hover:border-primary/30 transition-all group"
                  >
                    <div className="flex items-center gap-3">
                       <Badge variant="outline" className="font-mono text-[10px] font-black bg-background/50 group-hover:border-primary/50">
                         {pos.symbol.replace("CRYPTO:", "")}
                       </Badge>
                       <span className="text-[9px] font-black text-muted-foreground uppercase tracking-widest opacity-60">{pos.asset_class}</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="font-mono text-[11px] font-bold text-muted-foreground">${(pos.notional / 1000).toFixed(1)}k</span>
                      <Badge variant={tone} className="font-black text-[11px] min-w-[5ch] justify-center">
                        {pos.weight_pct.toFixed(1)}%
                      </Badge>
                    </div>
                  </div>
                );
              })}
            {heat.positions.length === 0 && (
              <div className="py-12 text-center text-[11px] font-black text-muted-foreground uppercase tracking-widest opacity-30 mt-12">
                Zero Exposure Active
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
