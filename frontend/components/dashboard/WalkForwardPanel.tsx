"use client";

import { useEffect, useState } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface PeriodMetrics {
  period: string;
  trades: number;
  wins: number;
  win_rate: number;
  avg_pnl_pct: number;
  sharpe: number;
  baseline_sharpe: number;
  max_dd_pct: number;
  avg_slippage_bps: number;
  regime_counts: Record<string, number>;
  component_alpha: Record<string, number>;
  gross_pnl_usd: number;
}

interface RegimePeriod {
  regime: string;
  period: string;
  trades: number;
  wins: number;
  win_rate: number;
  avg_pnl_pct: number;
}

interface WalkForwardReport {
  periods: PeriodMetrics[];
  regime_trend: RegimePeriod[];
  component_alpha_trend: Record<string, { period: string; alpha: number }[]>;
  regime_distribution: Record<string, number>;
  overall: {
    total_trades: number;
    total_wins: number;
    win_rate: number;
    avg_pnl_pct: number;
    sharpe: number;
    baseline_sharpe: number;
    total_periods: number;
    gross_pnl_usd: number;
  };
  generated_at: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt(n: number, decimals = 2): string {
  return Number.isFinite(n) ? n.toFixed(decimals) : "—";
}

function fmtPct(n: number): string {
  return `${Number.isFinite(n) ? (n * 100).toFixed(2) : "—"}%`;
}

function fmtMoney(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return `$${n >= 0 ? "+" : ""}${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

const REGIME_COLORS: Record<string, string> = {
  bull: "text-emerald-400",
  bear: "text-red-400",
  strong_bull: "text-emerald-300",
  strong_bear: "text-red-300",
  volatile: "text-amber-400",
  crisis: "text-orange-500",
  neutral: "text-slate-400",
  unknown: "text-slate-500",
};

const COMP_COLORS: Record<string, string> = {
  ml: "bg-violet-500",
  tech: "bg-blue-500",
  sentiment: "bg-amber-500",
  momentum: "bg-teal-500",
  pairs: "bg-pink-500",
};

function SharpeBar({ value, baseline }: { value: number; baseline: number }) {
  const max = Math.max(Math.abs(value), Math.abs(baseline), 0.1);
  const vPct = Math.min(100, (Math.abs(value) / max) * 100);
  const bPct = Math.min(100, (Math.abs(baseline) / max) * 100);
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className="w-20 text-xs text-muted-foreground">Strategy</span>
        <div className="flex-1 h-2 rounded bg-muted">
          <div
            className={`h-2 rounded ${value >= 0 ? "bg-emerald-500" : "bg-red-500"}`}
            style={{ width: `${vPct}%` }}
          />
        </div>
        <span className={`w-10 text-right text-xs font-mono ${value >= 0 ? "text-emerald-400" : "text-red-400"}`}>
          {fmt(value)}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-20 text-xs text-muted-foreground">Baseline</span>
        <div className="flex-1 h-2 rounded bg-muted">
          <div
            className={`h-2 rounded ${baseline >= 0 ? "bg-slate-400" : "bg-slate-600"}`}
            style={{ width: `${bPct}%` }}
          />
        </div>
        <span className="w-10 text-right text-xs font-mono text-muted-foreground">
          {fmt(baseline)}
        </span>
      </div>
    </div>
  );
}

function ComponentAlphaBar({ alpha, component }: { alpha: number; component: string }) {
  const color = COMP_COLORS[component] ?? "bg-slate-500";
  const pct = Math.min(100, Math.abs(alpha) * 500); // scale for visibility
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 text-xs text-muted-foreground capitalize">{component}</span>
      <div className="flex-1 h-2 rounded bg-muted">
        <div
          className={`h-2 rounded ${alpha >= 0 ? color : "bg-red-600"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`w-14 text-right text-xs font-mono ${alpha >= 0 ? "text-emerald-400" : "text-red-400"}`}>
        {(alpha * 100).toFixed(3)}%
      </span>
    </div>
  );
}

// ── Main Panel ────────────────────────────────────────────────────────────────

export default function WalkForwardPanel() {
  const [report, setReport] = useState<WalkForwardReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch("/api/v1/walk-forward", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!cancelled) setReport(data);
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const id = setInterval(load, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
        Loading walk-forward report…
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="flex items-center justify-center h-48 text-red-400 text-sm">
        {error || "No data available"}
      </div>
    );
  }

  const { overall, periods, regime_trend, component_alpha_trend, regime_distribution } = report;
  const sortedPeriods = [...periods].sort((a, b) => a.period.localeCompare(b.period));

  // Regime distribution total
  const regimeTotal = Object.values(regime_distribution).reduce((s, v) => s + v, 0);

  // Latest period component alpha
  const latestPeriod = sortedPeriods[sortedPeriods.length - 1];

  return (
    <div className="space-y-6">
      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Walk-Forward Validation</h2>
        <span className="text-xs text-muted-foreground">
          Updated {new Date(report.generated_at).toLocaleTimeString()}
        </span>
      </div>

      {/* ── Overall Summary ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "Total Trades", value: overall.total_trades.toString() },
          { label: "Win Rate", value: fmtPct(overall.win_rate) },
          { label: "Avg P&L", value: `${(overall.avg_pnl_pct * 100).toFixed(3)}%` },
          { label: "Gross P&L", value: fmtMoney(overall.gross_pnl_usd) },
        ].map(({ label, value }) => (
          <div key={label} className="rounded-xl border border-border/60 bg-background/50 p-3">
            <div className="text-xs text-muted-foreground mb-1">{label}</div>
            <div className="text-lg font-bold font-mono">{value}</div>
          </div>
        ))}
      </div>

      {/* ── Rolling Sharpe vs Baseline ── */}
      <div className="rounded-xl border border-border/60 bg-background/50 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold">Rolling Sharpe (Annualised)</h3>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-1 rounded bg-emerald-500" /> Strategy
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-1 rounded bg-slate-400" /> 50/50 Baseline
            </span>
          </div>
        </div>
        <div className="space-y-4">
          {sortedPeriods.map((p) => (
            <div key={p.period} className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground">{p.period}</div>
              <SharpeBar value={p.sharpe} baseline={p.baseline_sharpe} />
            </div>
          ))}
          {sortedPeriods.length === 0 && (
            <div className="text-xs text-muted-foreground italic">No periods with sufficient data</div>
          )}
        </div>
      </div>

      {/* ── Per-Period Table ── */}
      <div className="rounded-xl border border-border/60 bg-background/50 p-4 space-y-3">
        <h3 className="text-sm font-semibold">Period Summary</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-muted-foreground border-b border-border/40">
                <th className="text-left py-1 pr-3">Period</th>
                <th className="text-right py-1 pr-3">Trades</th>
                <th className="text-right py-1 pr-3">Win %</th>
                <th className="text-right py-1 pr-3">Avg P&L</th>
                <th className="text-right py-1 pr-3">Sharpe</th>
                <th className="text-right py-1 pr-3">MaxDD</th>
                <th className="text-right py-1">Slip bps</th>
              </tr>
            </thead>
            <tbody>
              {sortedPeriods.map((p) => (
                <tr key={p.period} className="border-b border-border/20 hover:bg-secondary/20">
                  <td className="py-1 pr-3 font-medium">{p.period}</td>
                  <td className="text-right pr-3">{p.trades}</td>
                  <td className={`text-right pr-3 ${p.win_rate >= 0.5 ? "text-emerald-400" : "text-red-400"}`}>
                    {fmtPct(p.win_rate)}
                  </td>
                  <td className={`text-right pr-3 ${p.avg_pnl_pct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {(p.avg_pnl_pct * 100).toFixed(3)}%
                  </td>
                  <td className={`text-right pr-3 font-mono ${p.sharpe >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {fmt(p.sharpe)}
                  </td>
                  <td className="text-right pr-3 text-amber-400">{fmt(p.max_dd_pct)}%</td>
                  <td className="text-right">{fmt(p.avg_slippage_bps, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Regime Win-Rate Trend ── */}
      <div className="rounded-xl border border-border/60 bg-background/50 p-4 space-y-3">
        <h3 className="text-sm font-semibold">Per-Regime Win Rate by Period</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-muted-foreground border-b border-border/40">
                <th className="text-left py-1 pr-3">Regime</th>
                <th className="text-left py-1 pr-3">Period</th>
                <th className="text-right py-1 pr-3">Trades</th>
                <th className="text-right py-1 pr-3">Win %</th>
                <th className="text-right py-1">Avg P&L</th>
              </tr>
            </thead>
            <tbody>
              {regime_trend.map((r, i) => (
                <tr key={`${r.regime}-${r.period}-${i}`} className="border-b border-border/20 hover:bg-secondary/20">
                  <td className={`py-1 pr-3 font-medium capitalize ${REGIME_COLORS[r.regime] ?? "text-slate-400"}`}>
                    {r.regime}
                  </td>
                  <td className="pr-3">{r.period}</td>
                  <td className="text-right pr-3">{r.trades}</td>
                  <td className={`text-right pr-3 ${r.win_rate >= 0.5 ? "text-emerald-400" : "text-red-400"}`}>
                    {fmtPct(r.win_rate)}
                  </td>
                  <td className={`text-right ${r.avg_pnl_pct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {(r.avg_pnl_pct * 100).toFixed(3)}%
                  </td>
                </tr>
              ))}
              {regime_trend.length === 0 && (
                <tr>
                  <td colSpan={5} className="text-center py-2 text-muted-foreground italic">No regime data</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Component Alpha (latest period) ── */}
      {latestPeriod && (
        <div className="rounded-xl border border-border/60 bg-background/50 p-4 space-y-3">
          <h3 className="text-sm font-semibold">
            Signal Component Alpha — {latestPeriod.period}
            <span className="ml-2 text-xs font-normal text-muted-foreground">(mean P&L × component weight)</span>
          </h3>
          <div className="space-y-2">
            {Object.entries(latestPeriod.component_alpha)
              .sort(([, a], [, b]) => b - a)
              .map(([comp, alpha]) => (
                <ComponentAlphaBar key={comp} component={comp} alpha={alpha} />
              ))}
          </div>
        </div>
      )}

      {/* ── Regime Distribution ── */}
      <div className="rounded-xl border border-border/60 bg-background/50 p-4 space-y-3">
        <h3 className="text-sm font-semibold">Regime Time Distribution (All Trades)</h3>
        <div className="space-y-2">
          {Object.entries(regime_distribution)
            .sort(([, a], [, b]) => b - a)
            .map(([regime, count]) => {
              const pct = regimeTotal > 0 ? (count / regimeTotal) * 100 : 0;
              return (
                <div key={regime} className="flex items-center gap-2">
                  <span className={`w-24 text-xs capitalize ${REGIME_COLORS[regime] ?? "text-slate-400"}`}>
                    {regime}
                  </span>
                  <div className="flex-1 h-3 rounded bg-muted">
                    <div
                      className={`h-3 rounded ${COMP_COLORS.ml}`}
                      style={{
                        width: `${pct}%`,
                        backgroundColor: regime === "bull" ? "#10b981"
                          : regime === "bear" ? "#ef4444"
                            : regime === "volatile" ? "#f59e0b"
                              : regime === "crisis" ? "#f97316"
                                : "#64748b"
                      }}
                    />
                  </div>
                  <span className="w-12 text-right text-xs font-mono text-muted-foreground">
                    {count} ({pct.toFixed(0)}%)
                  </span>
                </div>
              );
            })}
          {regimeTotal === 0 && (
            <div className="text-xs text-muted-foreground italic">No data</div>
          )}
        </div>
      </div>

      {/* ── Overall Sharpe summary ── */}
      <div className="rounded-xl border border-border/60 bg-background/50 p-4">
        <h3 className="text-sm font-semibold mb-3">Full-Period Sharpe Summary</h3>
        <SharpeBar value={overall.sharpe} baseline={overall.baseline_sharpe} />
        <div className="mt-2 text-xs text-muted-foreground">
          Alpha over baseline: {fmt(overall.sharpe - overall.baseline_sharpe)} Sharpe points
        </div>
      </div>
    </div>
  );
}
