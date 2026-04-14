"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";

interface PostMortem {
  symbol: string;
  pnl_pct: number;
  hold_hours: number;
  exit_reason: string;
  signal_quality: string;
  timing: string;
  regime_alignment: string;
  execution_drag: string;
  verdict: string;
  primary_failure: string;
  confidence_at_entry: number;
  signal_at_entry: number;
  slippage_bps: number;
  regime: string;
  timestamp: string;
}

interface Summary {
  total: number;
  win_rate: number;
  avg_pnl_pct: number;
  verdict_counts: Record<string, number>;
  failure_counts: Record<string, number>;
}

interface PostMortemData {
  recent: PostMortem[];
  summary: Summary;
  note?: string | null;
}

function categoryBadge(value: string) {
  if (value === "good") return "text-green-400";
  if (value === "bad") return "text-red-400";
  return "text-yellow-400";
}

function verdictColor(v: string) {
  if (v === "winner") return "text-green-400";
  if (v === "loser") return "text-red-400";
  return "text-yellow-400";
}

function failureLabel(f: string): string {
  const map: Record<string, string> = {
    none: "—",
    weak_signal: "Weak Signal",
    low_confidence: "Low Conf",
    bad_regime: "Bad Regime",
    premature_exit: "Early Exit",
    held_too_long: "Too Long",
    stop_hit: "Stop Hit",
    max_hold_expired: "Max Hold",
    slippage: "Slippage",
    unknown: "Unknown",
  };
  return map[f] ?? f;
}

export default function TradePostMortemPanel() {
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<PostMortemData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState("");

  const fetchData = async () => {
    try {
      const res = await fetch("/api/v1/trade-postmortem?n=20", {
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
    const id = setInterval(fetchData, 15_000);
    return () => clearInterval(id);
  }, [token]);

  if (error)
    return (
      <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
        Post-Mortem error: {error}
      </div>
    );

  if (!data)
    return (
      <div className="p-6 text-sm text-muted-foreground animate-pulse">
        Loading trade post-mortems…
      </div>
    );

  const { recent, summary } = data;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-foreground">Trade Post-Mortem</h2>
        <span className="text-xs text-muted-foreground">Updated {lastUpdated} · auto 15s</span>
      </div>

      {/* Summary row */}
      {summary && summary.total > 0 && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
            <div className="text-xs text-muted-foreground">Total Analyzed</div>
            <div className="text-lg font-bold font-mono">{summary.total}</div>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
            <div className="text-xs text-muted-foreground">Win Rate</div>
            <div className={`text-lg font-bold font-mono ${(summary.win_rate ?? 0) >= 0.5 ? "text-green-400" : "text-red-400"}`}>
              {((summary.win_rate ?? 0) * 100).toFixed(1)}%
            </div>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
            <div className="text-xs text-muted-foreground">Avg P&L</div>
            <div className={`text-lg font-bold font-mono ${(summary.avg_pnl_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
              {((summary.avg_pnl_pct ?? 0) * 100).toFixed(2)}%
            </div>
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
            <div className="text-xs text-muted-foreground">Top Failure</div>
            <div className="text-sm font-bold text-yellow-400">
              {Object.entries(summary.failure_counts ?? {})
                .filter(([k]) => k !== "none")
                .sort(([, a], [, b]) => b - a)[0]
                ? failureLabel(
                    Object.entries(summary.failure_counts ?? {})
                      .filter(([k]) => k !== "none")
                      .sort(([, a], [, b]) => b - a)[0][0]
                  )
                : "—"}
            </div>
          </div>
        </div>
      )}

      {/* Verdict counts */}
      {summary && summary.verdict_counts && Object.keys(summary.verdict_counts).length > 0 && (
        <div className="flex gap-3">
          {Object.entries(summary.verdict_counts).map(([verdict, count]) => (
            <div key={verdict} className="rounded-lg border border-border/40 bg-background/40 px-3 py-2 text-xs">
              <span className={`font-bold ${verdictColor(verdict)}`}>{count}</span>
              <span className="ml-1 text-muted-foreground capitalize">{verdict}s</span>
            </div>
          ))}
        </div>
      )}

      {/* Recent post-mortems table */}
      {recent.length === 0 ? (
        <p className="text-sm text-muted-foreground">No post-mortems yet — trade closes populate this.</p>
      ) : (
        <div className="rounded-2xl border border-border/60 bg-background/60 overflow-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border/40 text-muted-foreground">
                <th className="px-3 py-2 text-left">Symbol</th>
                <th className="px-3 py-2 text-right">P&L</th>
                <th className="px-3 py-2 text-right">Hold</th>
                <th className="px-3 py-2 text-center">Signal</th>
                <th className="px-3 py-2 text-center">Timing</th>
                <th className="px-3 py-2 text-center">Regime</th>
                <th className="px-3 py-2 text-center">Exec</th>
                <th className="px-3 py-2 text-center">Verdict</th>
                <th className="px-3 py-2 text-left">Failure</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((pm, i) => (
                <tr key={`${pm.symbol}-${i}`} className="border-b border-border/20 last:border-0 hover:bg-muted/10">
                  <td className="px-3 py-1.5 font-mono font-semibold">{pm.symbol}</td>
                  <td className={`px-3 py-1.5 text-right font-mono font-semibold ${pm.pnl_pct >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {pm.pnl_pct >= 0 ? "+" : ""}{(pm.pnl_pct * 100).toFixed(2)}%
                  </td>
                  <td className="px-3 py-1.5 text-right text-muted-foreground font-mono">
                    {pm.hold_hours.toFixed(1)}h
                  </td>
                  <td className={`px-3 py-1.5 text-center ${categoryBadge(pm.signal_quality)}`}>
                    {pm.signal_quality}
                  </td>
                  <td className={`px-3 py-1.5 text-center ${categoryBadge(pm.timing)}`}>
                    {pm.timing}
                  </td>
                  <td className={`px-3 py-1.5 text-center ${categoryBadge(pm.regime_alignment)}`}>
                    {pm.regime_alignment}
                  </td>
                  <td className={`px-3 py-1.5 text-center ${categoryBadge(pm.execution_drag)}`}>
                    {pm.execution_drag}
                  </td>
                  <td className={`px-3 py-1.5 text-center font-semibold ${verdictColor(pm.verdict)}`}>
                    {pm.verdict}
                  </td>
                  <td className="px-3 py-1.5 text-yellow-300/80">
                    {failureLabel(pm.primary_failure)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
