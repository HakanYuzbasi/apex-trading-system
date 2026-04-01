"use client";

import { useEffect, useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface RegimeStat {
  trade_count: number;
  win_rate_pct: number;
  avg_pnl_pct: number;
  total_pnl_pct: number;
  avg_hold_hours: number | null;
  sharpe_annualised: number;
  best_trade: { symbol: string; pnl_pct: number } | null;
  worst_trade: { symbol: string; pnl_pct: number } | null;
}

interface BacktestResult {
  regime_breakdown: Record<string, RegimeStat>;
  overall: RegimeStat;
  date_filter: string | null;
  generated_at: string;
  error?: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function pct(v: number | null | undefined, digits = 1): string {
  if (v == null) return "—";
  return (v * 100).toFixed(digits) + "%";
}

function pctRaw(v: number | null | undefined, digits = 1): string {
  if (v == null) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(digits) + "%";
}

function sharpeColor(v: number): string {
  if (v >= 1.5) return "text-green-600 dark:text-green-400";
  if (v >= 0.5) return "text-emerald-600 dark:text-emerald-400";
  if (v >= 0) return "text-gray-500";
  return "text-red-500 dark:text-red-400";
}

function winRateColor(v: number): string {
  if (v >= 60) return "text-green-600 dark:text-green-400";
  if (v >= 45) return "text-amber-600 dark:text-amber-400";
  return "text-red-500 dark:text-red-400";
}

const REGIME_ORDER = ["bull", "strong_bull", "neutral", "bear", "strong_bear", "volatile", "crisis"];

// ── Component ──────────────────────────────────────────────────────────────────

export default function RegimeBacktestPanel() {
  const [data, setData] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dateInput, setDateInput] = useState("");
  const [dateFilter, setDateFilter] = useState("");

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const url = dateFilter
        ? `/api/v1/regime-backtest?date=${dateFilter}`
        : `/api/v1/regime-backtest`;
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (json.error === "no_trades") {
        setData(null);
        setError("No completed trades found for this filter.");
        return;
      }
      setData(json);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load backtest");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [dateFilter]);

  const regimes = data
    ? REGIME_ORDER.filter((r) => r in (data.regime_breakdown ?? {})).concat(
        Object.keys(data.regime_breakdown ?? {}).filter(
          (r) => !REGIME_ORDER.includes(r)
        )
      )
    : [];

  if (loading && !data)
    return (
      <div className="p-6 text-sm text-gray-500 dark:text-gray-400">
        Loading regime backtest…
      </div>
    );

  return (
    <div className="space-y-6 p-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">Regime-Conditional Backtest</h2>
        <div className="flex items-center gap-2">
          <input
            type="date"
            value={dateInput}
            onChange={(e) => setDateInput(e.target.value)}
            className="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm"
          />
          <button
            onClick={() => setDateFilter(dateInput)}
            className="rounded bg-blue-500 px-3 py-1 text-xs text-white hover:bg-blue-600"
          >
            Filter
          </button>
          {dateFilter && (
            <button
              onClick={() => { setDateFilter(""); setDateInput(""); }}
              className="text-xs text-gray-500 hover:underline"
            >
              Clear
            </button>
          )}
          <button
            onClick={load}
            disabled={loading}
            className="text-xs text-blue-500 hover:underline disabled:opacity-50"
          >
            {loading ? "loading…" : "refresh"}
          </button>
        </div>
      </div>

      {error && (
        <div className="text-sm text-amber-600 dark:text-amber-400">{error}</div>
      )}

      {data && (
        <>
          {/* Overall summary */}
          {data.overall && (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { label: "Total Trades", value: String(data.overall.trade_count) },
                {
                  label: "Win Rate",
                  value: pct(data.overall.win_rate_pct / 100),
                  color: winRateColor(data.overall.win_rate_pct),
                },
                {
                  label: "Avg P&L",
                  value: pctRaw(data.overall.avg_pnl_pct / 100),
                  color: data.overall.avg_pnl_pct >= 0
                    ? "text-green-600 dark:text-green-400"
                    : "text-red-500",
                },
                {
                  label: "Sharpe",
                  value: data.overall.sharpe_annualised.toFixed(2),
                  color: sharpeColor(data.overall.sharpe_annualised),
                },
              ].map(({ label, value, color }) => (
                <div
                  key={label}
                  className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 text-center"
                >
                  <div className="text-xs text-gray-500 dark:text-gray-400">{label}</div>
                  <div className={`mt-1 text-xl font-bold ${color ?? ""}`}>{value}</div>
                </div>
              ))}
            </div>
          )}

          {/* Per-regime table */}
          {regimes.length > 0 && (
            <div>
              <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
                Per-Regime Breakdown
              </h3>
              <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 dark:bg-gray-800">
                    <tr>
                      {["Regime", "Trades", "Win Rate", "Avg P&L", "Total P&L", "Sharpe", "Avg Hold"].map(
                        (h) => (
                          <th
                            key={h}
                            className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400"
                          >
                            {h}
                          </th>
                        )
                      )}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                    {regimes.map((regime) => {
                      const s = data.regime_breakdown[regime];
                      return (
                        <tr
                          key={regime}
                          className="hover:bg-gray-50 dark:hover:bg-gray-800/50"
                        >
                          <td className="px-4 py-2 font-mono text-xs capitalize">{regime}</td>
                          <td className="px-4 py-2">{s.trade_count}</td>
                          <td className={`px-4 py-2 font-semibold ${winRateColor(s.win_rate_pct)}`}>
                            {s.win_rate_pct.toFixed(1)}%
                          </td>
                          <td
                            className={`px-4 py-2 ${s.avg_pnl_pct >= 0 ? "text-green-600 dark:text-green-400" : "text-red-500"}`}
                          >
                            {pctRaw(s.avg_pnl_pct / 100, 2)}
                          </td>
                          <td
                            className={`px-4 py-2 ${s.total_pnl_pct >= 0 ? "text-green-600 dark:text-green-400" : "text-red-500"}`}
                          >
                            {pctRaw(s.total_pnl_pct / 100, 2)}
                          </td>
                          <td className={`px-4 py-2 font-semibold ${sharpeColor(s.sharpe_annualised)}`}>
                            {s.sharpe_annualised.toFixed(2)}
                          </td>
                          <td className="px-4 py-2 text-gray-500">
                            {s.avg_hold_hours != null
                              ? s.avg_hold_hours < 1
                                ? `${(s.avg_hold_hours * 60).toFixed(0)}m`
                                : `${s.avg_hold_hours.toFixed(1)}h`
                              : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {!data && !loading && !error && (
        <p className="text-sm text-gray-400 dark:text-gray-500 text-center py-8">
          No trade data available. Backtest results will appear once trades are completed.
        </p>
      )}
    </div>
  );
}
