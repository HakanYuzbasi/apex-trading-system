"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";

interface SourceBucket {
  trades: number;
  wins: number;
  total_pnl: number;
  win_rate: number;
  avg_net_pnl: number;
  avg_pnl_bps: number;
  avg_holding_hours: number;
}

interface AttributionSummary {
  lookback_days: number;
  closed_trades: number;
  gross_pnl: number;
  net_pnl: number;
  commissions: number;
  modeled_execution_drag: number;
  modeled_slippage_drag: number;
  by_sleeve: Record<string, { trades: number; net_pnl: number; avg_net_pnl: number }>;
  by_asset_class: Record<string, { trades: number; net_pnl: number; avg_net_pnl: number }>;
}

interface SignalSources {
  lookback_days: number;
  by_signal_source: Record<string, SourceBucket>;
}

interface AttributionData {
  summary: AttributionSummary;
  signal_sources: SignalSources;
  note?: string | null;
}

const SOURCE_LABELS: Record<string, string> = {
  ml: "ML Model",
  technical: "Technical",
  sentiment: "Sentiment",
  cs_momentum: "CS Momentum",
  composite: "Composite",
};

function pnlColor(pnl: number) {
  return pnl >= 0 ? "text-green-400" : "text-red-400";
}

function winRateBar(wr: number) {
  const pct = Math.round(wr * 100);
  const color = pct >= 55 ? "bg-green-500" : pct >= 45 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-20 rounded-full bg-secondary/40">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-xs">{pct}%</span>
    </div>
  );
}

export default function AttributionPanel() {
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<AttributionData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState("");
  const [lookback, setLookback] = useState(30);

  const fetchData = async (lb = lookback) => {
    try {
      const res = await fetch(`/api/v1/attribution-summary?lookback_days=${lb}`, {
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
    fetchData(lookback);
    const id = setInterval(() => fetchData(lookback), 20_000);
    return () => clearInterval(id);
  }, [token, lookback]);

  if (error)
    return (
      <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
        Attribution error: {error}
      </div>
    );

  if (!data)
    return (
      <div className="p-6 text-sm text-muted-foreground animate-pulse">Loading attribution…</div>
    );

  const { summary, signal_sources } = data;
  const sources = signal_sources?.by_signal_source ?? {};

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-foreground">Performance Attribution</h2>
        <div className="flex items-center gap-3">
          <select
            value={lookback}
            onChange={(e) => setLookback(Number(e.target.value))}
            className="rounded-lg border border-border/60 bg-background px-2 py-1 text-xs text-foreground"
          >
            {[7, 14, 30, 60, 90].map((d) => (
              <option key={d} value={d}>{d}d</option>
            ))}
          </select>
          <span className="text-xs text-muted-foreground">Updated {lastUpdated} · auto 20s</span>
        </div>
      </div>

      {/* Summary row */}
      {summary && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
            <div className="text-xs text-muted-foreground">Closed Trades</div>
            <div className="text-lg font-bold font-mono">{summary.closed_trades ?? 0}</div>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
            <div className="text-xs text-muted-foreground">Net P&L</div>
            <div className={`text-lg font-bold font-mono ${pnlColor(summary.net_pnl ?? 0)}`}>
              ${(summary.net_pnl ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
            <div className="text-xs text-muted-foreground">Commissions</div>
            <div className="text-lg font-bold font-mono text-red-400/80">
              ${(summary.commissions ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
            <div className="text-xs text-muted-foreground">Slippage Drag</div>
            <div className="text-lg font-bold font-mono text-yellow-400/80">
              ${(summary.modeled_slippage_drag ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>
        </div>
      )}

      {/* Signal source breakdown */}
      {Object.keys(sources).length > 0 ? (
        <div className="rounded-2xl border border-border/60 bg-background/60 p-4">
          <h3 className="mb-3 text-sm font-semibold text-muted-foreground uppercase tracking-wide">
            P&L by Signal Source
          </h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-muted-foreground border-b border-border/40">
                <th className="pb-2 text-left">Source</th>
                <th className="pb-2 text-right">Trades</th>
                <th className="pb-2 text-left pl-4">Win Rate</th>
                <th className="pb-2 text-right">Avg P&L</th>
                <th className="pb-2 text-right">Avg bps</th>
                <th className="pb-2 text-right">Avg Hold</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(sources)
                .sort(([, a], [, b]) => b.total_pnl - a.total_pnl)
                .map(([source, bucket]) => (
                  <tr key={source} className="border-b border-border/20 last:border-0">
                    <td className="py-2 font-semibold">
                      {SOURCE_LABELS[source] ?? source}
                    </td>
                    <td className="py-2 text-right font-mono text-muted-foreground">
                      {bucket.trades}
                    </td>
                    <td className="py-2 pl-4">{winRateBar(bucket.win_rate)}</td>
                    <td className={`py-2 text-right font-mono font-semibold ${pnlColor(bucket.avg_net_pnl)}`}>
                      ${bucket.avg_net_pnl.toFixed(0)}
                    </td>
                    <td className={`py-2 text-right font-mono ${pnlColor(bucket.avg_pnl_bps)}`}>
                      {bucket.avg_pnl_bps.toFixed(0)}
                    </td>
                    <td className="py-2 text-right font-mono text-muted-foreground">
                      {bucket.avg_holding_hours.toFixed(1)}h
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          No attribution data yet — closed trades populate this.
        </p>
      )}

      {/* By asset class */}
      {summary?.by_asset_class && Object.keys(summary.by_asset_class).length > 0 && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
          {Object.entries(summary.by_asset_class).map(([cls, bucket]) => (
            <div key={cls} className="rounded-2xl border border-border/60 bg-background/60 p-3">
              <div className="text-xs text-muted-foreground uppercase">{cls}</div>
              <div className={`text-base font-bold font-mono ${pnlColor(bucket.net_pnl)}`}>
                ${bucket.net_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </div>
              <div className="text-xs text-muted-foreground">{bucket.trades} trades</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
