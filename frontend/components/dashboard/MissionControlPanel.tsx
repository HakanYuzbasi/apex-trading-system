"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";

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

function tierColor(tier: string) {
  if (tier === "RED") return "text-red-400";
  if (tier === "YELLOW") return "text-yellow-400";
  return "text-green-400";
}

function regimeColor(regime: string) {
  const r = regime.toLowerCase();
  if (r.includes("bear") || r.includes("crisis") || r.includes("volatile")) return "text-red-400";
  if (r.includes("bull")) return "text-green-400";
  return "text-yellow-400";
}

function Bar({ pct, color, label, value }: { pct: number; color: string; label: string; value: string }) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{label}</span>
        <span className="font-mono font-semibold text-foreground">{value}</span>
      </div>
      <div className="h-2 w-full rounded-full bg-secondary/40">
        <div className={`h-2 rounded-full transition-all ${color}`} style={{ width: `${clamped}%` }} />
      </div>
    </div>
  );
}

function ProbGauge({ prob }: { prob: number | null }) {
  if (prob === null) return <span className="text-muted-foreground text-sm">—</span>;
  const pct = Math.round(prob * 100);
  const color = pct >= 70 ? "bg-red-500" : pct >= 45 ? "bg-yellow-500" : "bg-green-500";
  return (
    <div className="flex items-center gap-2">
      <div className="h-3 w-32 rounded-full bg-secondary/40">
        <div className={`h-3 rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-sm font-semibold">{pct}%</span>
    </div>
  );
}

function Pill({ value, green, yellow }: { value: number; green: number; yellow: number }) {
  const cls = value >= green ? "bg-green-500/20 text-green-400" : value >= yellow ? "bg-yellow-500/20 text-yellow-400" : "bg-red-500/20 text-red-400";
  return <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${cls}`}>{(value * 100).toFixed(0)}%</span>;
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
      <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
        Mission Control error: {error}
      </div>
    );

  if (!data)
    return <div className="p-6 text-sm text-muted-foreground animate-pulse">Loading mission control…</div>;

  const { system, risk_budget, top_positions, predictive } = data;

  const dailyBarColor =
    risk_budget.daily_pnl >= 0 ? "bg-green-500" : risk_budget.daily_pnl_pct < -3 ? "bg-red-500" : "bg-yellow-500";
  const ddBarColor = risk_budget.drawdown_pct > 5 ? "bg-red-500" : risk_budget.drawdown_pct > 2 ? "bg-yellow-500" : "bg-green-500/60";
  const marginUsedPct = system.equity > 0 ? (risk_budget.active_margin / system.equity) * 100 : 0;

  const fmtPnl = (v: number) => `${v >= 0 ? "+" : ""}$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-foreground">Mission Control</h2>
        <span className="text-xs text-muted-foreground">Updated {lastUpdated} · auto 10s</span>
      </div>

      {/* ── Row 1: System Status + Risk Budget ── */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">

        {/* System Status */}
        <div className="rounded-2xl border border-border/60 bg-background/60 p-4 space-y-3">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">System Status</h3>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-xs text-muted-foreground">Regime</div>
              <div className={`text-base font-bold capitalize ${regimeColor(system.regime)}`}>{system.regime}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">VIX</div>
              <div className={`text-base font-bold font-mono ${system.vix >= 30 ? "text-red-400" : system.vix >= 20 ? "text-yellow-400" : "text-green-400"}`}>
                {system.vix > 0 ? system.vix.toFixed(1) : "—"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Governor</div>
              <div className={`text-base font-bold ${tierColor(system.governor_tier)}`}>{system.governor_tier}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Kill Switch</div>
              <div className={`text-base font-bold ${system.kill_switch_active ? "text-red-500" : "text-green-400"}`}>
                {system.kill_switch_active ? "ACTIVE" : "OFF"}
              </div>
            </div>
          </div>

          <div className="border-t border-border/40 pt-2 grid grid-cols-3 gap-2">
            <div>
              <div className="text-xs text-muted-foreground">Equity</div>
              <div className="text-base font-bold font-mono">${system.equity.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Meta Conf</div>
              <div className="flex items-center gap-1 mt-0.5">
                <Pill value={system.meta_confidence} green={0.8} yellow={0.6} />
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Survival</div>
              <div className="flex items-center gap-1 mt-0.5">
                <Pill value={system.survival_probability} green={0.9} yellow={0.7} />
              </div>
            </div>
          </div>
        </div>

        {/* Risk Budget */}
        <div className="rounded-2xl border border-border/60 bg-background/60 p-4 space-y-3">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Risk Budget</h3>

          <Bar
            pct={Math.abs(risk_budget.daily_pnl_pct) * 20}
            color={dailyBarColor}
            label="Daily P&L"
            value={`${fmtPnl(risk_budget.daily_pnl)} (${risk_budget.daily_pnl >= 0 ? "+" : ""}${risk_budget.daily_pnl_pct.toFixed(2)}%)`}
          />
          <Bar
            pct={risk_budget.drawdown_pct * 10}
            color={ddBarColor}
            label="Max Drawdown"
            value={`${risk_budget.drawdown_pct.toFixed(2)}%`}
          />
          <Bar
            pct={marginUsedPct}
            color={marginUsedPct > 80 ? "bg-red-500" : marginUsedPct > 50 ? "bg-yellow-500" : "bg-blue-500"}
            label="Margin Used"
            value={`$${risk_budget.active_margin.toLocaleString("en-US", { maximumFractionDigits: 0 })}`}
          />
          <Bar
            pct={risk_budget.positions_pct}
            color={risk_budget.positions_pct >= 80 ? "bg-yellow-500" : "bg-indigo-500"}
            label="Positions"
            value={`${risk_budget.position_count} / ${risk_budget.max_positions}`}
          />

          <div className="grid grid-cols-2 gap-2 pt-1 border-t border-border/40">
            <div>
              <div className="text-xs text-muted-foreground">Realized P&L</div>
              <div className={`text-sm font-bold font-mono ${risk_budget.realized_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                {fmtPnl(risk_budget.realized_pnl)}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Unrealized P&L</div>
              <div className={`text-sm font-bold font-mono ${risk_budget.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                {fmtPnl(risk_budget.unrealized_pnl)}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Row 2: Top Positions + Predictive ── */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">

        {/* Top Positions */}
        <div className="rounded-2xl border border-border/60 bg-background/60 p-4">
          <h3 className="mb-3 text-sm font-semibold text-muted-foreground uppercase tracking-wide">
            Top Positions (by |P&amp;L|)
          </h3>
          {top_positions.length === 0 ? (
            <p className="text-sm text-muted-foreground">No open positions</p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-muted-foreground border-b border-border/40">
                  <th className="pb-1 text-left">Symbol</th>
                  <th className="pb-1 text-right">P&amp;L</th>
                  <th className="pb-1 text-right">P&amp;L%</th>
                  <th className="pb-1 text-right">Signal</th>
                </tr>
              </thead>
              <tbody>
                {top_positions.map((p) => (
                  <tr key={p.symbol} className="border-b border-border/20 last:border-0">
                    <td className="py-1 font-mono text-xs font-semibold">{p.symbol.replace("CRYPTO:", "")}</td>
                    <td className={`py-1 text-right font-mono text-xs ${p.pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                      {fmtPnl(p.pnl)}
                    </td>
                    <td className={`py-1 text-right font-mono text-xs font-semibold ${p.pnl_pct >= 0 ? "text-green-400" : "text-red-400"}`}>
                      {p.pnl_pct >= 0 ? "+" : ""}{p.pnl_pct.toFixed(2)}%
                    </td>
                    <td className={`py-1 text-right text-xs ${
                      p.signal_direction === "bullish" ? "text-green-400"
                        : p.signal_direction === "bearish" ? "text-red-400"
                        : "text-muted-foreground"
                    }`}>
                      {p.signal_direction ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Predictive Indicators */}
        <div className="rounded-2xl border border-border/60 bg-background/60 p-4 space-y-4">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Predictive Indicators</h3>

          <div>
            <div className="text-xs text-muted-foreground mb-1">
              Regime Transition Probability
              {predictive.transition_direction && predictive.transition_direction !== "unknown" && (
                <span className="ml-1 text-yellow-400">→ {predictive.transition_direction}</span>
              )}
            </div>
            <ProbGauge prob={predictive.transition_probability} />
          </div>

          {predictive.transition_size_mult !== null && predictive.transition_size_mult < 1 && (
            <div className="rounded-lg bg-yellow-500/10 border border-yellow-500/30 px-3 py-2 text-xs text-yellow-300">
              Size dampened to {((predictive.transition_size_mult ?? 1) * 100).toFixed(0)}% — transition risk elevated
            </div>
          )}

          <div className="grid grid-cols-2 gap-3 border-t border-border/40 pt-3">
            <div>
              <div className="text-xs text-muted-foreground">RL Updates</div>
              <div className="text-sm font-bold font-mono">{predictive.rl_total_updates?.toLocaleString() ?? "—"}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">RL Epsilon</div>
              <div className="text-sm font-bold font-mono">
                {predictive.rl_epsilon !== null ? `${(predictive.rl_epsilon * 100).toFixed(1)}%` : "—"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Bayesian Vol</div>
              <div className="text-sm font-bold font-mono">
                {predictive.bayesian_vol_prob !== null ? `${((predictive.bayesian_vol_prob ?? 0) * 100).toFixed(1)}%` : "—"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Universe Scored</div>
              <div className="text-sm font-bold font-mono">
                {predictive.universe_scored !== null ? `${predictive.universe_scored} syms` : "—"}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
