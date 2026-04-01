"use client";

import { useEffect, useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface FactorBucket {
  factor: string;
  pnl_sum: number;
  pnl_pct_sum: number;
  avg_pnl_pct: number;
  win_rate: number;
  trade_count: number;
}

interface AssetMap {
  [factor: string]: FactorBucket;
}

interface FactorPnlReport {
  generated_at: string;
  lookback_days: number;
  total_trades: number;
  total_pnl: number;
  total_pnl_pct: number;
  by_factor: FactorBucket[];
  by_asset: Record<string, AssetMap>;
  by_regime: Record<string, AssetMap>;
  last_1d: FactorBucket[];
  last_7d: FactorBucket[];
  last_30d: FactorBucket[];
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function pct(v: number | null | undefined): string {
  if (v == null) return "—";
  return (v * 100).toFixed(2) + "%";
}

function pctSign(v: number | null | undefined): string {
  if (v == null) return "—";
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
}

function dollar(v: number | null | undefined): string {
  if (v == null) return "—";
  return (v >= 0 ? "+" : "") + "$" + Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 });
}

function pnlColor(v: number): string {
  if (v > 0) return "text-green-600 dark:text-green-400";
  if (v < 0) return "text-red-500 dark:text-red-400";
  return "text-gray-500";
}

const FACTOR_LABELS: Record<string, string> = {
  ml: "ML Model",
  technical: "Technical",
  sentiment: "Sentiment",
  momentum: "Momentum",
  residual: "Residual",
};

const FACTOR_COLORS: Record<string, string> = {
  ml: "bg-blue-500",
  technical: "bg-purple-500",
  sentiment: "bg-amber-500",
  momentum: "bg-emerald-500",
  residual: "bg-gray-400",
};

// ── Component ──────────────────────────────────────────────────────────────────

export default function FactorPnlPanel() {
  const [data, setData] = useState<FactorPnlReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lookback, setLookback] = useState(7);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/v1/factor-pnl?lookback_days=${lookback}`, {
        cache: "no-store",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load factor P&L");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 60_000);
    return () => clearInterval(id);
  }, [lookback]);

  if (loading && !data)
    return (
      <div className="p-6 text-sm text-gray-500 dark:text-gray-400">
        Loading factor P&L…
      </div>
    );

  if (error)
    return (
      <div className="p-6 text-sm text-red-500">
        {error} —{" "}
        <button onClick={load} className="underline">
          retry
        </button>
      </div>
    );

  if (!data) return null;

  // Stacked-bar total for normalising bar widths
  const totalAbsPnl = data.by_factor.reduce(
    (s, b) => s + Math.abs(b.pnl_pct_sum),
    0
  );

  return (
    <div className="space-y-6 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Factor P&amp;L Decomposition</h2>
        <div className="flex items-center gap-3">
          <select
            value={lookback}
            onChange={(e) => setLookback(Number(e.target.value))}
            className="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm"
          >
            {[1, 7, 14, 30].map((d) => (
              <option key={d} value={d}>
                {d}d
              </option>
            ))}
          </select>
          <button
            onClick={load}
            disabled={loading}
            className="text-xs text-blue-500 hover:underline disabled:opacity-50"
          >
            {loading ? "refreshing…" : "refresh"}
          </button>
        </div>
      </div>

      {/* Summary row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: "Total P&L", value: dollar(data.total_pnl), color: pnlColor(data.total_pnl) },
          { label: "P&L %", value: pctSign(data.total_pnl_pct), color: pnlColor(data.total_pnl_pct) },
          { label: "Trades", value: String(data.total_trades), color: "" },
          { label: "Lookback", value: `${lookback}d`, color: "" },
        ].map(({ label, value, color }) => (
          <div
            key={label}
            className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 text-center"
          >
            <div className="text-xs text-gray-500 dark:text-gray-400">{label}</div>
            <div className={`mt-1 text-xl font-bold ${color}`}>{value}</div>
          </div>
        ))}
      </div>

      {/* Factor breakdown table */}
      {data.by_factor.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
            P&amp;L by Factor
          </h3>
          <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  {["Factor", "P&L %", "Avg/Trade", "Win Rate", "Trades", "Contribution"].map(
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
                {data.by_factor.map((b) => {
                  const barWidth =
                    totalAbsPnl > 0
                      ? Math.round((Math.abs(b.pnl_pct_sum) / totalAbsPnl) * 100)
                      : 0;
                  return (
                    <tr
                      key={b.factor}
                      className="hover:bg-gray-50 dark:hover:bg-gray-800/50"
                    >
                      <td className="px-4 py-2 font-medium">
                        <div className="flex items-center gap-2">
                          <span
                            className={`inline-block h-2.5 w-2.5 rounded-full ${
                              FACTOR_COLORS[b.factor] ?? "bg-gray-400"
                            }`}
                          />
                          {FACTOR_LABELS[b.factor] ?? b.factor}
                        </div>
                      </td>
                      <td className={`px-4 py-2 font-semibold ${pnlColor(b.pnl_pct_sum)}`}>
                        {pctSign(b.pnl_pct_sum)}
                      </td>
                      <td className={`px-4 py-2 ${pnlColor(b.avg_pnl_pct)}`}>
                        {pctSign(b.avg_pnl_pct)}
                      </td>
                      <td className="px-4 py-2 text-gray-600 dark:text-gray-400">
                        {pct(b.win_rate)}
                      </td>
                      <td className="px-4 py-2 text-gray-500">{b.trade_count}</td>
                      <td className="px-4 py-2 w-32">
                        <div className="h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
                          <div
                            className={`h-2 rounded-full ${
                              b.pnl_pct_sum >= 0 ? "bg-green-500" : "bg-red-500"
                            }`}
                            style={{ width: `${barWidth}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* By asset class */}
      {Object.keys(data.by_asset).length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
            By Asset Class
          </h3>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(data.by_asset).map(([asset, fmap]) => {
              const dominant = Object.values(fmap).sort(
                (a, b) => Math.abs(b.pnl_pct_sum) - Math.abs(a.pnl_pct_sum)
              )[0];
              const totalPnl = Object.values(fmap).reduce(
                (s, b) => s + b.pnl_pct_sum,
                0
              );
              return (
                <div
                  key={asset}
                  className="rounded-lg border border-gray-200 dark:border-gray-700 p-3"
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-sm">{asset}</span>
                    <span className={`text-sm font-bold ${pnlColor(totalPnl)}`}>
                      {pctSign(totalPnl)}
                    </span>
                  </div>
                  {dominant && (
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Dominant:{" "}
                      <span className="font-medium text-gray-700 dark:text-gray-300">
                        {FACTOR_LABELS[dominant.factor] ?? dominant.factor}
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Empty state */}
      {data.total_trades === 0 && (
        <p className="text-sm text-gray-400 dark:text-gray-500 text-center py-8">
          No closed trades in the last {lookback} days.
          <br />
          Factor decomposition will populate once trades complete.
        </p>
      )}
    </div>
  );
}
