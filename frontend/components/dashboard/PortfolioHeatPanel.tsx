"use client";

import { useEffect, useState } from "react";

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

function heatColor(weightPct: number): string {
  if (weightPct >= 20) return "bg-red-500/20 text-red-300";
  if (weightPct >= 12) return "bg-orange-500/20 text-orange-300";
  if (weightPct >= 6) return "bg-yellow-500/20 text-yellow-300";
  return "bg-green-500/10 text-green-300";
}

function driftBadge(health: string): string {
  if (health === "critical") return "bg-red-600 text-white";
  if (health === "degrading") return "bg-orange-500 text-white";
  return "bg-emerald-600 text-white";
}

function hhiLabel(hhi: number): { label: string; color: string } {
  if (hhi > 0.40) return { label: "Concentrated", color: "text-red-400" };
  if (hhi > 0.20) return { label: "Moderate", color: "text-yellow-400" };
  return { label: "Diversified", color: "text-emerald-400" };
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
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    const interval = setInterval(load, 30_000); // refresh every 30s
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (loading) {
    return (
      <div className="rounded-lg border border-white/10 bg-white/5 p-4">
        <h2 className="text-sm font-semibold text-white/70 mb-3">Portfolio Heat</h2>
        <p className="text-xs text-white/40 animate-pulse">Loading…</p>
      </div>
    );
  }

  if (error || !heat) {
    return (
      <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-4">
        <h2 className="text-sm font-semibold text-red-400 mb-1">Portfolio Heat</h2>
        <p className="text-xs text-red-300/60">{error ?? "No data"}</p>
      </div>
    );
  }

  const hhiInfo = hhiLabel(heat.hhi_concentration);

  return (
    <div className="rounded-lg border border-white/10 bg-white/5 p-4 space-y-4">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Portfolio Heat</h2>
        <div className="flex gap-2 text-xs">
          <span className="px-2 py-0.5 rounded bg-white/10 text-white/60">
            Regime: <span className="text-white font-medium">{heat.regime}</span>
          </span>
          <span className="px-2 py-0.5 rounded bg-white/10 text-white/60">
            VIX: <span className="text-white font-medium">{heat.vix?.toFixed(1) ?? "–"}</span>
          </span>
        </div>
      </div>

      {/* Summary row */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="rounded bg-white/5 p-2 text-center">
          <div className="text-white/50">Positions</div>
          <div className="text-white font-bold text-lg">{heat.position_count}</div>
        </div>
        <div className="rounded bg-white/5 p-2 text-center">
          <div className="text-white/50">Notional</div>
          <div className="text-white font-bold text-lg">
            ${(heat.total_notional / 1000).toFixed(1)}k
          </div>
        </div>
        <div className="rounded bg-white/5 p-2 text-center">
          <div className="text-white/50">HHI</div>
          <div className={`font-bold text-lg ${hhiInfo.color}`}>
            {heat.hhi_concentration.toFixed(2)}
          </div>
          <div className={`text-[10px] ${hhiInfo.color}`}>{hhiInfo.label}</div>
        </div>
      </div>

      {/* Asset class breakdown */}
      {Object.keys(heat.by_asset_class).length > 0 && (
        <div className="space-y-1">
          <div className="text-[11px] text-white/40 uppercase tracking-wider mb-1">By Asset Class</div>
          {Object.entries(heat.by_asset_class).map(([ac, info]) => (
            <div key={ac} className="flex items-center gap-2 text-xs">
              <span className="w-14 text-white/60 capitalize">{ac}</span>
              <div className="flex-1 h-1.5 rounded bg-white/10 overflow-hidden">
                <div
                  className={`h-full rounded ${ac === "crypto" ? "bg-blue-400" : "bg-emerald-400"}`}
                  style={{ width: `${info.weight_pct}%` }}
                />
              </div>
              <span className="w-10 text-right text-white/60">{info.weight_pct.toFixed(0)}%</span>
              <span className="w-12 text-right text-white/40">${(info.notional / 1000).toFixed(1)}k</span>
            </div>
          ))}
        </div>
      )}

      {/* Position heatmap table */}
      {heat.positions.length > 0 && (
        <div className="space-y-1">
          <div className="text-[11px] text-white/40 uppercase tracking-wider mb-1">Position Weights</div>
          <div className="space-y-0.5 max-h-48 overflow-y-auto pr-1">
            {[...heat.positions]
              .sort((a, b) => b.weight_pct - a.weight_pct)
              .map((pos) => (
                <div
                  key={pos.symbol}
                  className={`flex items-center justify-between px-2 py-1 rounded text-xs ${heatColor(pos.weight_pct)}`}
                >
                  <span className="font-mono">{pos.symbol.replace("CRYPTO:", "")}</span>
                  <span className="text-[10px] opacity-70 ml-2">{pos.asset_class}</span>
                  <div className="flex gap-3 ml-auto">
                    <span>${(pos.notional / 1000).toFixed(1)}k</span>
                    <span className="font-semibold">{pos.weight_pct.toFixed(1)}%</span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Alpha decay hint */}
      {heat.alpha_decay && (
        <div className="flex gap-4 text-xs text-white/50 border-t border-white/10 pt-2">
          <span>
            Optimal hold:{" "}
            <span className="text-white/80">
              {heat.alpha_decay.optimal_hold_hours?.toFixed(1) ?? "–"}h
            </span>
          </span>
          {heat.alpha_decay.alpha_half_life != null && (
            <span>
              α half-life:{" "}
              <span className="text-white/80">
                {heat.alpha_decay.alpha_half_life.toFixed(1)}h
              </span>
            </span>
          )}
        </div>
      )}

      {/* Model drift badge */}
      {heat.model_drift && (
        <div className="flex items-center gap-2 text-xs border-t border-white/10 pt-2">
          <span
            className={`px-2 py-0.5 rounded text-[10px] font-semibold ${driftBadge(heat.model_drift.health)}`}
          >
            Model: {heat.model_drift.health.toUpperCase()}
          </span>
          <span className="text-white/40">
            IC {heat.model_drift.ic_current.toFixed(3)} · HR{" "}
            {(heat.model_drift.hit_rate_current * 100).toFixed(0)}%
          </span>
          {heat.model_drift.should_retrain && (
            <span className="ml-auto text-orange-400 font-semibold animate-pulse">
              ⚠ Retrain
            </span>
          )}
        </div>
      )}
    </div>
  );
}
