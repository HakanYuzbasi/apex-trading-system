"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/lib/auth-context";

interface TcaSummary {
  closed_trades: number;
  win_rate_pct: number;
  total_net_pnl: number;
  total_execution_drag: number;
  alpha_before_costs: number;
  cost_ratio_pct: number;
  total_fills: number;
  total_rejections: number;
  rejection_breakdown: Record<string, number>;
  execution_health_score: number;
}

interface TcaSymbolRow {
  symbol: string;
  closed_trades: number;
  win_rate_pct: number | null;
  net_pnl: number;
  execution_drag: number;
  avg_entry_slip_bps: number | null;
  avg_exit_slip_bps: number | null;
  median_fill_ms: number | null;
  p95_fill_ms: number | null;
  fills: number;
  exit_reasons: Record<string, number>;
  open_position: boolean;
  rejections: Record<string, number>;
}

interface TcaReport {
  available: boolean;
  note?: string;
  generated_at?: string;
  summary?: TcaSummary;
  per_symbol?: TcaSymbolRow[];
}

function fmt2(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return "—";
  return v.toFixed(2);
}

function fmtPct(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return "—";
  return v.toFixed(1) + "%";
}

function fmtMs(v: number | null | undefined): string {
  if (v == null) return "—";
  return v.toFixed(0) + "ms";
}

function fmtUsd(v: number): string {
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "+";
  return `${sign}$${abs.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
}

function HealthGauge({ score }: { score: number }) {
  const color =
    score >= 75 ? "text-green-400" : score >= 50 ? "text-yellow-400" : "text-red-400";
  const bg =
    score >= 75 ? "bg-green-400/20" : score >= 50 ? "bg-yellow-400/20" : "bg-red-400/20";
  return (
    <div className={`flex items-center gap-3 rounded-xl border border-border/60 ${bg} px-4 py-3`}>
      <div>
        <p className="text-[11px] text-muted-foreground uppercase tracking-wider">Execution Health</p>
        <p className={`text-3xl font-bold font-mono mt-0.5 ${color}`}>
          {score.toFixed(1)}<span className="text-base font-normal text-muted-foreground ml-1">/100</span>
        </p>
      </div>
    </div>
  );
}

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="flex flex-col gap-0.5 rounded-lg border border-border/60 bg-background/50 px-3 py-2.5">
      <span className="text-[11px] text-muted-foreground uppercase tracking-wide">{label}</span>
      <span className={`text-base font-semibold font-mono ${color ?? "text-foreground"}`}>{value}</span>
      {sub && <span className="text-[10px] text-muted-foreground/60">{sub}</span>}
    </div>
  );
}

export default function TcaReportPanel() {
  const { token } = useAuthContext();
  const [data, setData] = useState<TcaReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch("/api/v1/tca-report", {
          headers: token ? { authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const id = setInterval(load, 120_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [token]);

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading TCA report…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available || !data.summary) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "TCA report unavailable — engine may not be running."}
      </div>
    );
  }

  const s = data.summary;
  const rows = data.per_symbol ?? [];

  const rejBreakdown = Object.entries(s.rejection_breakdown ?? {})
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="p-4 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-foreground">Transaction Cost Analysis</h2>
        <span className="text-[11px] text-muted-foreground font-mono">
          {data.generated_at ? data.generated_at.slice(0, 19) + " UTC" : ""}
        </span>
      </div>

      {/* Health + top KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
        <HealthGauge score={s.execution_health_score} />
        <StatCard
          label="Net P&L"
          value={fmtUsd(s.total_net_pnl)}
          sub={`${s.closed_trades} closed trades`}
          color={s.total_net_pnl >= 0 ? "text-green-400" : "text-red-400"}
        />
        <StatCard
          label="Win Rate"
          value={fmtPct(s.win_rate_pct)}
          color={s.win_rate_pct >= 55 ? "text-green-400" : s.win_rate_pct >= 45 ? "text-yellow-400" : "text-red-400"}
        />
        <StatCard
          label="Exec Drag"
          value={fmtUsd(s.total_execution_drag)}
          sub={`${fmtPct(s.cost_ratio_pct)} of gross alpha`}
          color="text-orange-400"
        />
      </div>

      {/* Secondary stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <StatCard label="Gross Alpha" value={fmtUsd(s.alpha_before_costs)} sub="Before costs" />
        <StatCard label="Total Fills" value={String(s.total_fills)} />
        <StatCard
          label="Rejections"
          value={String(s.total_rejections)}
          color={s.total_rejections > 20 ? "text-red-400" : "text-foreground"}
        />
        <StatCard label="Cost Ratio" value={fmtPct(s.cost_ratio_pct)} sub="Of gross alpha" />
      </div>

      {/* Rejection breakdown */}
      {rejBreakdown.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Rejection Breakdown
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {rejBreakdown.map(([reason, count]) => (
              <div
                key={reason}
                className="flex items-center justify-between rounded-lg border border-border/60 bg-background/50 px-3 py-2"
              >
                <span className="text-xs text-muted-foreground capitalize">
                  {reason.replace(/_/g, " ")}
                </span>
                <span className="text-sm font-semibold font-mono text-foreground">{count}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Per-symbol table */}
      {rows.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Per-Symbol Execution Quality
          </h3>
          <div className="overflow-x-auto rounded-lg border border-border/60">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-border/60 text-muted-foreground text-[11px] uppercase">
                  <th className="px-3 py-2 text-left">Symbol</th>
                  <th className="px-3 py-2 text-right">Trades</th>
                  <th className="px-3 py-2 text-right">WR%</th>
                  <th className="px-3 py-2 text-right">Net P&L</th>
                  <th className="px-3 py-2 text-right">Drag</th>
                  <th className="px-3 py-2 text-right">eSlip</th>
                  <th className="px-3 py-2 text-right">xSlip</th>
                  <th className="px-3 py-2 text-right">Med Fill</th>
                  <th className="px-3 py-2 text-right">P95 Fill</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const pnlColor = row.net_pnl > 0 ? "text-green-400" : row.net_pnl < 0 ? "text-red-400" : "";
                  const wrColor =
                    row.win_rate_pct == null
                      ? ""
                      : row.win_rate_pct >= 55
                      ? "text-green-400"
                      : row.win_rate_pct < 45
                      ? "text-red-400"
                      : "text-yellow-400";
                  const slipWarn =
                    (row.avg_exit_slip_bps ?? 0) > 80 ? "text-orange-400" : "";
                  return (
                    <tr
                      key={row.symbol}
                      className="border-b border-border/40 hover:bg-secondary/30 transition-colors"
                    >
                      <td className="px-3 py-1.5 text-left text-foreground font-semibold">
                        {row.symbol}
                        {row.open_position && (
                          <span className="ml-1 text-[9px] text-blue-400 uppercase">live</span>
                        )}
                      </td>
                      <td className="px-3 py-1.5 text-right text-muted-foreground">{row.closed_trades}</td>
                      <td className={`px-3 py-1.5 text-right ${wrColor}`}>{fmtPct(row.win_rate_pct)}</td>
                      <td className={`px-3 py-1.5 text-right ${pnlColor}`}>{fmtUsd(row.net_pnl)}</td>
                      <td className="px-3 py-1.5 text-right text-orange-400">{fmt2(row.execution_drag)}</td>
                      <td className="px-3 py-1.5 text-right text-muted-foreground">
                        {row.avg_entry_slip_bps != null ? row.avg_entry_slip_bps.toFixed(1) + "bp" : "—"}
                      </td>
                      <td className={`px-3 py-1.5 text-right ${slipWarn || "text-muted-foreground"}`}>
                        {row.avg_exit_slip_bps != null ? row.avg_exit_slip_bps.toFixed(1) + "bp" : "—"}
                      </td>
                      <td className="px-3 py-1.5 text-right text-muted-foreground">{fmtMs(row.median_fill_ms)}</td>
                      <td className="px-3 py-1.5 text-right text-muted-foreground">{fmtMs(row.p95_fill_ms)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
