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
  bull: "text-positive",
  bear: "text-negative",
  strong_bull: "text-positive/80",
  strong_bear: "text-negative/80",
  volatile: "text-warning",
  crisis: "text-negative font-black",
  neutral: "text-muted-foreground/60",
  unknown: "text-muted-foreground/40",
};

const COMP_COLORS: Record<string, string> = {
  ml: "bg-primary",
  tech: "bg-primary/60",
  sentiment: "bg-warning",
  momentum: "bg-positive",
  pairs: "bg-primary/40",
};

function SharpeBar({ value, baseline }: { value: number; baseline: number }) {
  const max = Math.max(Math.abs(value), Math.abs(baseline), 0.1);
  const vPct = Math.min(100, (Math.abs(value) / max) * 100);
  const bPct = Math.min(100, (Math.abs(baseline) / max) * 100);
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className="w-20 text-[10px] font-black uppercase tracking-tight text-muted-foreground/60">Strategy</span>
        <div className="flex-1 h-2 rounded bg-muted/40 overflow-hidden">
          <div
            className={`h-2 rounded ${value >= 0 ? "bg-positive" : "bg-negative"}`}
            style={{ width: `${vPct}%` }}
          />
        </div>
        <span className={`w-10 text-right text-[10px] font-black font-mono ${value >= 0 ? "text-positive" : "text-negative"}`}>
          {fmt(value)}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-20 text-[10px] font-black uppercase tracking-tight text-muted-foreground/40">Baseline</span>
        <div className="flex-1 h-2 rounded bg-muted/40 overflow-hidden">
          <div
            className={`h-2 rounded ${baseline >= 0 ? "bg-muted-foreground/30" : "bg-muted-foreground/10"}`}
            style={{ width: `${bPct}%` }}
          />
        </div>
        <span className="w-10 text-right text-[10px] font-black font-mono text-muted-foreground/40">
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
      <span className="w-20 text-[10px] font-black uppercase tracking-tight text-muted-foreground/60">{component}</span>
      <div className="flex-1 h-2 rounded bg-muted/40 overflow-hidden">
        <div
          className={`h-2 rounded ${alpha >= 0 ? color : "bg-negative"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`w-14 text-right text-[10px] font-black font-mono ${alpha >= 0 ? "text-positive" : "text-negative"}`}>
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
      <div className="flex items-center justify-between border-b border-border/10 pb-4">
        <div className="flex items-center gap-3">
          <div className="px-2 py-0.5 rounded-lg bg-primary/20 text-primary border border-primary/20">
             <span className="text-[10px] font-black uppercase tracking-widest">WALK-FORWARD VALIDATION</span>
          </div>
        </div>
        <span className="text-[10px] font-black uppercase tracking-tighter text-muted-foreground/40">
          UPDATED {new Date(report.generated_at).toLocaleTimeString().toUpperCase()}
        </span>
      </div>

      {/* ── Overall Summary ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "TOTAL TRADES", value: overall.total_trades.toString() },
          { label: "WIN RATE", value: fmtPct(overall.win_rate) },
          { label: "AVG P&L", value: `${(overall.avg_pnl_pct * 100).toFixed(3)}%` },
          { label: "GROSS P&L", value: fmtMoney(overall.gross_pnl_usd) },
        ].map(({ label, value }) => (
          <div key={label} className="glass-card rounded-xl border border-border/10 p-4">
            <div className="text-[9px] font-black uppercase tracking-widest text-muted-foreground/60 mb-2">{label}</div>
            <div className="text-xl font-black font-mono tracking-tight leading-none">{value}</div>
          </div>
        ))}
      </div>

      {/* ── Rolling Sharpe vs Baseline ── */}
      <div className="glass-card rounded-xl border border-border/10 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-[10px] font-black uppercase tracking-widest text-foreground">ROLLING SHARPE (ANNUALISED)</h3>
          <div className="flex items-center gap-4 text-[9px] font-black uppercase tracking-tighter text-muted-foreground/40">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 h-1 rounded bg-positive" /> STRATEGY
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 h-1 rounded bg-muted-foreground/30" /> BASELINE
            </span>
          </div>
        </div>
        <div className="space-y-4">
          {sortedPeriods.map((p) => (
            <div key={p.period} className="space-y-1.5">
              <div className="text-[10px] font-black uppercase tracking-tight text-muted-foreground/60">{p.period}</div>
              <SharpeBar value={p.sharpe} baseline={p.baseline_sharpe} />
            </div>
          ))}
          {sortedPeriods.length === 0 && (
            <div className="text-xs text-muted-foreground italic">No periods with sufficient data</div>
          )}
        </div>
      </div>

      {/* ── Per-Period Table ── */}
      <div className="glass-card rounded-xl border border-border/10 p-5 space-y-4">
        <h3 className="text-[10px] font-black uppercase tracking-widest text-foreground">PERIOD SUMMARY</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-muted-foreground border-b border-border/10">
                <th className="text-left py-2 font-black uppercase tracking-widest text-[9px]">PERIOD</th>
                <th className="text-right py-2 font-black uppercase tracking-widest text-[9px]">TRADES</th>
                <th className="text-right py-2 font-black uppercase tracking-widest text-[9px]">WIN %</th>
                <th className="text-right py-2 font-black uppercase tracking-widest text-[9px]">AVG P&L</th>
                <th className="text-right py-2 font-black uppercase tracking-widest text-[9px]">SHARPE</th>
                <th className="text-right py-2 font-black uppercase tracking-widest text-[9px]">MAXDD</th>
                <th className="text-right py-2 font-black uppercase tracking-widest text-[9px]">SLIP BPS</th>
              </tr>
            </thead>
            <tbody>
              {sortedPeriods.map((p) => (
                <tr key={p.period} className="border-b border-border/5 hover:bg-white/5 transition-colors">
                  <td className="py-2 font-black font-mono tracking-tight text-[11px]">{p.period}</td>
                  <td className="text-right py-2 text-muted-foreground font-mono text-[11px]">{p.trades}</td>
                  <td className={`text-right py-2 font-black font-mono text-[11px] ${p.win_rate >= 0.5 ? "text-positive" : "text-negative"}`}>
                    {fmtPct(p.win_rate)}
                  </td>
                  <td className={`text-right py-2 font-black font-mono text-[11px] ${p.avg_pnl_pct >= 0 ? "text-positive" : "text-negative"}`}>
                    {(p.avg_pnl_pct * 100).toFixed(3)}%
                  </td>
                  <td className={`text-right py-2 font-black font-mono text-[11px] ${p.sharpe >= 0 ? "text-positive" : "text-negative"}`}>
                    {fmt(p.sharpe)}
                  </td>
                  <td className="text-right py-2 text-warning font-black font-mono text-[11px]">{fmt(p.max_dd_pct)}%</td>
                  <td className="text-right py-2 text-muted-foreground/60 font-mono text-[11px]">{fmt(p.avg_slippage_bps, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Regime Win-Rate Trend ── */}
      <div className="glass-card rounded-xl border border-border/10 p-5 space-y-4">
        <h3 className="text-[10px] font-black uppercase tracking-widest text-foreground">PER-REGIME WIN RATE BY PERIOD</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-muted-foreground border-b border-border/10">
                <th className="text-left py-2 font-black uppercase tracking-widest text-[9px]">REGIME</th>
                <th className="text-left py-2 font-black uppercase tracking-widest text-[9px]">PERIOD</th>
                <th className="text-right py-2 font-black uppercase tracking-widest text-[9px]">TRADES</th>
                <th className="text-right py-2 font-black uppercase tracking-widest text-[9px]">WIN %</th>
                <th className="text-right py-2 font-black uppercase tracking-widest text-[9px]">AVG P&L</th>
              </tr>
            </thead>
            <tbody>
              {regime_trend.map((r, i) => (
                <tr key={`${r.regime}-${r.period}-${i}`} className="border-b border-border/5 hover:bg-white/5 transition-colors">
                  <td className={`py-2 font-black uppercase tracking-widest text-[10px] ${REGIME_COLORS[r.regime] ?? "text-muted-foreground/40"}`}>
                    {r.regime}
                  </td>
                  <td className="py-2 text-muted-foreground font-mono text-[11px]">{r.period}</td>
                  <td className="text-right py-2 text-muted-foreground/60 font-mono text-[11px]">{r.trades}</td>
                  <td className={`text-right py-2 font-black font-mono text-[11px] ${r.win_rate >= 0.5 ? "text-positive" : "text-negative"}`}>
                    {fmtPct(r.win_rate)}
                  </td>
                  <td className={`text-right py-2 font-black font-mono text-[11px] ${r.avg_pnl_pct >= 0 ? "text-positive" : "text-negative"}`}>
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
          <h3 className="text-[10px] font-black uppercase tracking-widest text-foreground">
          SIGNAL COMPONENT ALPHA — {latestPeriod.period}
          <span className="ml-3 text-[9px] font-medium text-muted-foreground opacity-40 lowercase">(mean P&L × component weight)</span>
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
      <div className="glass-card rounded-xl border border-border/10 p-5 space-y-4">
        <h3 className="text-[10px] font-black uppercase tracking-widest text-foreground">REGIME TIME DISTRIBUTION (ALL TRADES)</h3>
        <div className="space-y-3">
          {Object.entries(regime_distribution)
            .sort(([, a], [, b]) => b - a)
            .map(([regime, count]) => {
              const pct = regimeTotal > 0 ? (count / regimeTotal) * 100 : 0;
              return (
                <div key={regime} className="flex items-center gap-3">
                  <span className={`w-24 text-[10px] font-black uppercase tracking-widest ${REGIME_COLORS[regime] ?? "text-muted-foreground/40"}`}>
                    {regime}
                  </span>
                  <div className="flex-1 h-3 rounded-full bg-muted/40 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700 bg-primary/60"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: regime === "bull" ? "var(--positive)"
                          : regime === "bear" ? "var(--negative)"
                            : regime === "volatile" ? "var(--warning)"
                              : regime === "crisis" ? "var(--negative)"
                                : "var(--muted-foreground)"
                      }}
                    />
                  </div>
                  <span className="w-16 text-right text-[10px] font-black font-mono text-muted-foreground/60">
                    {count} <span className="text-[9px] opacity-40 ml-1">({pct.toFixed(0)}%)</span>
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
      <div className="glass-card rounded-xl border border-border/10 p-5">
        <h3 className="text-[10px] font-black uppercase tracking-widest text-foreground mb-4">FULL-PERIOD SHARPE SUMMARY</h3>
        <SharpeBar value={overall.sharpe} baseline={overall.baseline_sharpe} />
        <div className="mt-4 text-[9px] font-black uppercase tracking-[0.2em] text-muted-foreground/40">
          Alpha over baseline: <span className="text-positive">{fmt(overall.sharpe - overall.baseline_sharpe)}</span> Sharpe points
        </div>
      </div>
    </div>
  );
}
