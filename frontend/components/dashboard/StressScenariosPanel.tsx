"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/lib/auth-context";

interface StressScenario {
  scenario_id: string;
  scenario_name: string;
  scenario_type: string;
  portfolio_pnl: number;
  portfolio_return_pct: number;
  max_drawdown_pct: number;
  var_95_stressed: number;
  expected_shortfall: number;
  worst_positions: { symbol: string; pnl: number }[];
  breached_limits: string[];
  estimated_liquidation_cost: number;
  recommendations: string[];
}

interface StressData {
  available: boolean;
  note?: string;
  scenarios?: StressScenario[];
  capital?: number;
  n_positions?: number;
}

function fmtUsd(v: number): string {
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "+";
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}k`;
  return `${sign}$${abs.toFixed(0)}`;
}

function fmtPct(v: number): string {
  return (v >= 0 ? "+" : "") + v.toFixed(1) + "%";
}

function severityColor(returnPct: number): string {
  if (returnPct <= -20) return "text-red-500";
  if (returnPct <= -10) return "text-red-400";
  if (returnPct <= -5) return "text-orange-400";
  return "text-yellow-400";
}

function severityBg(returnPct: number): string {
  if (returnPct <= -20) return "border-red-500/40 bg-red-950/20";
  if (returnPct <= -10) return "border-red-400/30 bg-red-900/10";
  if (returnPct <= -5) return "border-orange-400/30 bg-orange-900/10";
  return "border-yellow-400/30 bg-yellow-900/10";
}

function ScenarioCard({ s }: { s: StressScenario }) {
  const [expanded, setExpanded] = useState(false);
  const pnlColor = severityColor(s.portfolio_return_pct);
  const bg = severityBg(s.portfolio_return_pct);

  return (
    <div className={`rounded-xl border ${bg} p-4 space-y-2`}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-foreground">{s.scenario_name}</h3>
          <span className="text-[11px] text-muted-foreground capitalize">
            {s.scenario_type.replace(/_/g, " ")}
          </span>
        </div>
        <div className="text-right shrink-0">
          <p className={`text-lg font-bold font-mono ${pnlColor}`}>
            {fmtPct(s.portfolio_return_pct)}
          </p>
          <p className={`text-sm font-mono ${pnlColor}`}>{fmtUsd(s.portfolio_pnl)}</p>
        </div>
      </div>

      {/* Key metrics row */}
      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <p className="text-[10px] text-muted-foreground uppercase">Max DD</p>
          <p className="text-xs font-mono text-red-400">{fmtPct(s.max_drawdown_pct)}</p>
        </div>
        <div>
          <p className="text-[10px] text-muted-foreground uppercase">VaR 95%</p>
          <p className="text-xs font-mono text-orange-400">{fmtUsd(s.var_95_stressed)}</p>
        </div>
        <div>
          <p className="text-[10px] text-muted-foreground uppercase">Liq Cost</p>
          <p className="text-xs font-mono text-muted-foreground">{fmtUsd(s.estimated_liquidation_cost)}</p>
        </div>
      </div>

      {/* Breached limits badges */}
      {s.breached_limits.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {s.breached_limits.map((limit) => (
            <span
              key={limit}
              className="text-[10px] px-1.5 py-0.5 rounded bg-red-900/30 text-red-300 border border-red-700/30"
            >
              {limit.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}

      {/* Expand for details */}
      <button
        type="button"
        onClick={() => setExpanded((x) => !x)}
        className="text-[11px] text-muted-foreground hover:text-foreground transition"
      >
        {expanded ? "▲ hide details" : "▼ show details"}
      </button>

      {expanded && (
        <div className="space-y-2 pt-1 border-t border-border/40">
          {s.worst_positions.length > 0 && (
            <div>
              <p className="text-[11px] font-semibold text-muted-foreground uppercase mb-1">Worst Positions</p>
              {s.worst_positions.map((wp) => (
                <div key={wp.symbol} className="flex justify-between text-xs font-mono">
                  <span className="text-foreground">{wp.symbol}</span>
                  <span className="text-red-400">{fmtUsd(wp.pnl)}</span>
                </div>
              ))}
            </div>
          )}
          {s.recommendations.length > 0 && (
            <div>
              <p className="text-[11px] font-semibold text-muted-foreground uppercase mb-1">Recommendations</p>
              <ul className="space-y-0.5">
                {s.recommendations.map((rec, i) => (
                  <li key={i} className="text-[11px] text-muted-foreground">• {rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function StressScenariosPanel() {
  const { token } = useAuthContext();
  const [data, setData] = useState<StressData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch("/api/v1/stress-scenarios", {
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
    const id = setInterval(load, 300_000); // refresh every 5 min
    return () => { cancelled = true; clearInterval(id); };
  }, [token]);

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Running stress scenarios…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available || !data.scenarios) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "Stress test unavailable — engine may not be running."}
      </div>
    );
  }

  const worstReturn = data.scenarios.reduce(
    (mn, s) => Math.min(mn, s.portfolio_return_pct), 0
  );

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-foreground">Portfolio Stress Scenarios</h2>
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground font-mono">
          <span>{data.n_positions ?? 0} positions</span>
          <span>•</span>
          <span>
            Worst: <span className="text-red-400">{fmtPct(worstReturn)}</span>
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {data.scenarios.map((s) => (
          <ScenarioCard key={s.scenario_id} s={s} />
        ))}
      </div>
    </div>
  );
}
