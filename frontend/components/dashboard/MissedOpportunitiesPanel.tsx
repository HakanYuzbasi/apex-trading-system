"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";

interface PendingOpp {
  symbol: string;
  signal_strength: number;
  confidence: number;
  direction: string;
  regime: string;
  filter_reason: string;
  entry_price: number;
  asset_class: string;
  signal_date: string;
}

interface MissedReport {
  total_missed: number;
  total_missed_pnl_5d: number;
  total_missed_pnl_10d: number;
  by_filter_reason: Record<string, number>;
  by_regime: Record<string, number>;
  top_missed_symbols: { symbol: string; total_missed_pnl_pct: number }[];
  generated_at: string;
}

interface MissedData {
  pending_count: number;
  completed_count: number;
  recent_pending: PendingOpp[];
  report: MissedReport;
  note?: string | null;
}

const FILTER_LABELS: Record<string, string> = {
  signal_threshold: "Signal Too Weak",
  confidence_threshold: "Low Confidence",
  regime: "Bad Regime",
  portfolio_heat: "Portfolio Heat",
  drawdown_gate: "Drawdown Gate",
  kill_switch: "Kill Switch",
  universe_score: "Universe Score",
};

function pnlColor(pnl: number) {
  return pnl >= 0.02 ? "text-green-400" : pnl >= 0 ? "text-green-300/70" : "text-red-400";
}

export default function MissedOpportunitiesPanel() {
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<MissedData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState("");

  const fetchData = async () => {
    try {
      const res = await fetch("/api/v1/missed-opportunities", {
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
    const id = setInterval(fetchData, 20_000);
    return () => clearInterval(id);
  }, [token]);

  if (error)
    return (
      <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
        Missed Opportunities error: {error}
      </div>
    );

  if (!data)
    return (
      <div className="p-6 text-sm text-muted-foreground animate-pulse">
        Loading missed opportunities…
      </div>
    );

  const { report, recent_pending } = data;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-foreground">Missed Opportunities</h2>
        <span className="text-xs text-muted-foreground">Updated {lastUpdated} · auto 20s</span>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
          <div className="text-xs text-muted-foreground">Pending Review</div>
          <div className="text-lg font-bold font-mono text-yellow-400">{data.pending_count}</div>
        </div>
        <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
          <div className="text-xs text-muted-foreground">Completed (20d)</div>
          <div className="text-lg font-bold font-mono">{data.completed_count}</div>
        </div>
        {report && (
          <>
            <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
              <div className="text-xs text-muted-foreground">Missed P&L (5d)</div>
              <div className={`text-lg font-bold font-mono ${pnlColor(report.total_missed_pnl_5d)}`}>
                {((report.total_missed_pnl_5d ?? 0) * 100).toFixed(1)}%
              </div>
            </div>
            <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
              <div className="text-xs text-muted-foreground">Missed P&L (10d)</div>
              <div className={`text-lg font-bold font-mono ${pnlColor(report.total_missed_pnl_10d)}`}>
                {((report.total_missed_pnl_10d ?? 0) * 100).toFixed(1)}%
              </div>
            </div>
          </>
        )}
      </div>

      {/* Filter reason breakdown */}
      {report?.by_filter_reason && Object.keys(report.by_filter_reason).length > 0 && (
        <div className="rounded-2xl border border-border/60 bg-background/60 p-4">
          <h3 className="mb-3 text-sm font-semibold text-muted-foreground uppercase tracking-wide">
            Filtered By Reason
          </h3>
          <div className="flex flex-wrap gap-2">
            {Object.entries(report.by_filter_reason)
              .sort(([, a], [, b]) => b - a)
              .map(([reason, count]) => (
                <div
                  key={reason}
                  className="rounded-lg border border-border/40 bg-secondary/20 px-3 py-1.5 text-xs"
                >
                  <span className="font-bold text-yellow-300">{count}</span>
                  <span className="ml-1 text-muted-foreground">
                    {FILTER_LABELS[reason] ?? reason}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Top missed symbols */}
      {report?.top_missed_symbols && report.top_missed_symbols.length > 0 && (
        <div className="rounded-2xl border border-border/60 bg-background/60 p-4">
          <h3 className="mb-3 text-sm font-semibold text-muted-foreground uppercase tracking-wide">
            Top Missed Symbols (by retrospective P&L)
          </h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-muted-foreground border-b border-border/40">
                <th className="pb-2 text-left">Symbol</th>
                <th className="pb-2 text-right">Missed P&L (10d)</th>
              </tr>
            </thead>
            <tbody>
              {report.top_missed_symbols.slice(0, 10).map((s) => (
                <tr key={s.symbol} className="border-b border-border/20 last:border-0">
                  <td className="py-1.5 font-mono font-semibold">{s.symbol}</td>
                  <td className={`py-1.5 text-right font-mono font-semibold ${pnlColor(s.total_missed_pnl_pct)}`}>
                    {s.total_missed_pnl_pct >= 0 ? "+" : ""}
                    {(s.total_missed_pnl_pct * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Recent pending queue */}
      {recent_pending.length > 0 && (
        <div className="rounded-2xl border border-border/60 bg-background/60 overflow-auto">
          <div className="px-4 pt-4 pb-2">
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              Recent Pending (awaiting retrospective pricing)
            </h3>
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border/40 text-muted-foreground px-4">
                <th className="px-3 py-2 text-left">Symbol</th>
                <th className="px-3 py-2 text-left">Reason</th>
                <th className="px-3 py-2 text-right">Signal</th>
                <th className="px-3 py-2 text-right">Conf</th>
                <th className="px-3 py-2 text-left">Regime</th>
                <th className="px-3 py-2 text-right">Price</th>
              </tr>
            </thead>
            <tbody>
              {recent_pending.slice(0, 15).map((o, i) => (
                <tr key={`${o.symbol}-${i}`} className="border-b border-border/20 last:border-0 hover:bg-muted/10">
                  <td className="px-3 py-1.5 font-mono font-semibold">{o.symbol}</td>
                  <td className="px-3 py-1.5 text-yellow-300/80">
                    {FILTER_LABELS[o.filter_reason] ?? o.filter_reason}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono">{o.signal_strength.toFixed(3)}</td>
                  <td className="px-3 py-1.5 text-right font-mono">{(o.confidence * 100).toFixed(0)}%</td>
                  <td className="px-3 py-1.5 text-muted-foreground capitalize">{o.regime}</td>
                  <td className="px-3 py-1.5 text-right font-mono">${o.entry_price.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {recent_pending.length === 0 && data.pending_count === 0 && (
        <p className="text-sm text-muted-foreground">
          No missed opportunities recorded yet — filtered signals populate this.
        </p>
      )}
    </div>
  );
}
