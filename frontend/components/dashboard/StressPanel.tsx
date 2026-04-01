"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/lib/auth-context";

interface ScenarioSummary {
  scenario_id: string;
  scenario_name: string;
  portfolio_pnl: number;
  portfolio_return: number;
  max_drawdown: number;
  breached_limits: string[];
  recommendations: string[];
  top_position_losses: { symbol: string; pnl: number }[];
}

interface StressState {
  enabled: boolean;
  active: boolean;
  evaluated_at: string;
  scenario_count: number;
  action: string;
  halt_new_entries: boolean;
  size_multiplier: number;
  reason: string;
  worst_scenario_id: string;
  worst_scenario_name: string;
  worst_portfolio_return: number;
  worst_portfolio_pnl: number;
  worst_drawdown: number;
  breached_limits: string[];
  recommendations: string[];
  scenarios: ScenarioSummary[];
}

interface StressData {
  state: StressState | null;
  note?: string | null;
}

function actionColor(action: string) {
  if (action === "halt") return "text-red-400";
  if (action === "warn") return "text-yellow-400";
  return "text-green-400";
}

function pctColor(pct: number) {
  if (pct < -0.05) return "text-red-400";
  if (pct < -0.02) return "text-yellow-400";
  return "text-green-400";
}

export default function StressPanel() {
  const { token } = useAuthContext();
  const [data, setData] = useState<StressData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState("");

  const fetchData = async () => {
    try {
      const res = await fetch("/api/v1/stress-state", {
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
        Stress Engine error: {error}
      </div>
    );

  if (!data)
    return (
      <div className="p-6 text-sm text-muted-foreground animate-pulse">Loading stress engine…</div>
    );

  const s = data.state;

  if (!s) {
    return (
      <div className="rounded-xl border border-border/40 bg-background/60 p-6 text-sm text-muted-foreground">
        {data.note ?? "Stress engine not available."}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-foreground">Intraday Stress Engine</h2>
        <span className="text-xs text-muted-foreground">Updated {lastUpdated} · auto 20s</span>
      </div>

      {/* Control state summary */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
          <div className="text-xs text-muted-foreground">Status</div>
          <div className={`text-base font-bold uppercase ${actionColor(s.action)}`}>
            {s.action}
          </div>
        </div>
        <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
          <div className="text-xs text-muted-foreground">New Entries</div>
          <div className={`text-base font-bold ${s.halt_new_entries ? "text-red-400" : "text-green-400"}`}>
            {s.halt_new_entries ? "HALTED" : "ALLOWED"}
          </div>
        </div>
        <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
          <div className="text-xs text-muted-foreground">Size Multiplier</div>
          <div className={`text-base font-bold font-mono ${s.size_multiplier < 0.75 ? "text-yellow-400" : "text-green-400"}`}>
            {(s.size_multiplier * 100).toFixed(0)}%
          </div>
        </div>
        <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
          <div className="text-xs text-muted-foreground">Scenarios</div>
          <div className="text-base font-bold font-mono">{s.scenario_count}</div>
        </div>
      </div>

      {/* Worst scenario */}
      {s.worst_scenario_name && (
        <div className="rounded-2xl border border-border/60 bg-background/60 p-4 space-y-2">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
            Worst Scenario
          </h3>
          <div className="flex flex-wrap gap-4">
            <div>
              <div className="text-xs text-muted-foreground">Name</div>
              <div className="text-sm font-semibold">{s.worst_scenario_name}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Portfolio Return</div>
              <div className={`text-sm font-bold font-mono ${pctColor(s.worst_portfolio_return)}`}>
                {(s.worst_portfolio_return * 100).toFixed(2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Max Drawdown</div>
              <div className={`text-sm font-bold font-mono ${pctColor(-s.worst_drawdown)}`}>
                {(s.worst_drawdown * 100).toFixed(2)}%
              </div>
            </div>
          </div>
          {s.breached_limits.length > 0 && (
            <div className="mt-2">
              <div className="text-xs text-muted-foreground mb-1">Breached Limits</div>
              <div className="flex flex-wrap gap-1">
                {s.breached_limits.map((l, i) => (
                  <span key={i} className="rounded-md bg-red-500/10 border border-red-500/30 px-2 py-0.5 text-xs text-red-300">
                    {l}
                  </span>
                ))}
              </div>
            </div>
          )}
          {s.recommendations.length > 0 && (
            <div className="mt-2">
              <div className="text-xs text-muted-foreground mb-1">Recommendations</div>
              <ul className="space-y-0.5">
                {s.recommendations.map((r, i) => (
                  <li key={i} className="text-xs text-yellow-300">• {r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Scenario table */}
      {s.scenarios.length > 0 && (
        <div className="rounded-2xl border border-border/60 bg-background/60 overflow-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border/40 text-muted-foreground">
                <th className="px-3 py-2 text-left">Scenario</th>
                <th className="px-3 py-2 text-right">Return</th>
                <th className="px-3 py-2 text-right">P&L</th>
                <th className="px-3 py-2 text-right">Drawdown</th>
                <th className="px-3 py-2 text-left">Breaches</th>
              </tr>
            </thead>
            <tbody>
              {s.scenarios.map((sc) => (
                <tr key={sc.scenario_id} className="border-b border-border/20 last:border-0">
                  <td className="px-3 py-1.5 font-semibold">{sc.scenario_name}</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${pctColor(sc.portfolio_return)}`}>
                    {(sc.portfolio_return * 100).toFixed(2)}%
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${sc.portfolio_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                    ${sc.portfolio_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${pctColor(-sc.max_drawdown)}`}>
                    {(sc.max_drawdown * 100).toFixed(2)}%
                  </td>
                  <td className="px-3 py-1.5 text-muted-foreground">
                    {sc.breached_limits.length > 0 ? sc.breached_limits.join(", ") : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {s.reason && (
        <p className="text-xs text-muted-foreground italic">{s.reason}</p>
      )}
    </div>
  );
}
