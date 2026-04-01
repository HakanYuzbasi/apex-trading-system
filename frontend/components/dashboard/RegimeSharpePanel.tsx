"use client";

import { useEffect, useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface GateInfo {
  fires: number;
  blocks: number;
  block_rate: number;
  total_decisions: number;
}

interface BlockedAnalysis {
  total_decisions: number;
  total_blocked: number;
  block_rate: number;
  by_first_gate: Record<string, number>;
}

interface BlockedSymbol {
  symbol: string;
  block_rate: number;
  blocked: number;
  total: number;
}

interface DiagnosticsReport {
  lookback_days: number;
  total_records: number;
  entered: number;
  completed_trades: number;
  overall_win_rate: number | null;
  gate_attribution: Record<string, GateInfo>;
  blocked_analysis: BlockedAnalysis;
  most_blocked_symbols: BlockedSymbol[];
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function pct(v: number | null | undefined): string {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function bar(ratio: number, width = 80): string {
  const filled = Math.round(Math.max(0, Math.min(1, ratio)) * width);
  return "█".repeat(filled) + "░".repeat(width - filled);
}

function gateColor(blockRate: number): string {
  if (blockRate >= 0.6) return "text-red-500 dark:text-red-400";
  if (blockRate >= 0.3) return "text-amber-500 dark:text-amber-400";
  return "text-green-600 dark:text-green-400";
}

// ── Component ────────────────────────────────────────────────────────────────

export default function RegimeSharpePanel() {
  const [data, setData] = useState<DiagnosticsReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lookback, setLookback] = useState(7);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/v1/regime-sharpe?lookback_days=${lookback}`,
        { cache: "no-store" }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (e: any) {
      setError(e.message ?? "Failed to load diagnostics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 60_000);
    return () => clearInterval(id);
  }, [lookback]);

  if (loading && !data) {
    return (
      <div className="p-6 text-sm text-gray-500 dark:text-gray-400">
        Loading diagnostics…
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-sm text-red-500">
        {error} —{" "}
        <button onClick={load} className="underline">
          retry
        </button>
      </div>
    );
  }

  if (!data) return null;

  const gates = Object.entries(data.gate_attribution ?? {});
  const topGates = gates.slice(0, 10);

  return (
    <div className="space-y-6 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Gate Diagnostics Leaderboard</h2>
        <div className="flex items-center gap-3">
          <select
            value={lookback}
            onChange={(e) => setLookback(Number(e.target.value))}
            className="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm"
          >
            {[1, 3, 7, 14, 30].map((d) => (
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
          {
            label: "Total Decisions",
            value: data.total_records.toLocaleString(),
          },
          { label: "Entries", value: data.entered.toLocaleString() },
          {
            label: "Block Rate",
            value: pct(data.blocked_analysis.block_rate),
          },
          {
            label: "Win Rate",
            value: data.overall_win_rate != null ? pct(data.overall_win_rate) : "—",
          },
        ].map(({ label, value }) => (
          <div
            key={label}
            className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 text-center"
          >
            <div className="text-xs text-gray-500 dark:text-gray-400">{label}</div>
            <div className="mt-1 text-xl font-bold">{value}</div>
          </div>
        ))}
      </div>

      {/* Gate attribution table */}
      {topGates.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
            Top Blocking Gates (last {lookback}d)
          </h3>
          <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  {["Gate", "Blocks", "Fires", "Block Rate", "Distribution"].map(
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
                {topGates.map(([gate, info]) => (
                  <tr
                    key={gate}
                    className="hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <td className="px-4 py-2 font-mono text-xs">{gate}</td>
                    <td className="px-4 py-2 font-semibold">{info.blocks}</td>
                    <td className="px-4 py-2 text-gray-500">{info.fires}</td>
                    <td
                      className={`px-4 py-2 font-semibold ${gateColor(
                        info.block_rate
                      )}`}
                    >
                      {pct(info.block_rate)}
                    </td>
                    <td className="px-4 py-2 font-mono text-[10px] text-gray-400">
                      {bar(info.block_rate, 20)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Most blocked symbols */}
      {data.most_blocked_symbols.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
            Most Blocked Symbols
          </h3>
          <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  {["Symbol", "Blocked", "Total", "Block Rate"].map((h) => (
                    <th
                      key={h}
                      className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                {data.most_blocked_symbols.map((row) => (
                  <tr
                    key={row.symbol}
                    className="hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <td className="px-4 py-2 font-semibold">{row.symbol}</td>
                    <td className="px-4 py-2">{row.blocked}</td>
                    <td className="px-4 py-2 text-gray-500">{row.total}</td>
                    <td
                      className={`px-4 py-2 font-semibold ${gateColor(
                        row.block_rate
                      )}`}
                    >
                      {pct(row.block_rate)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* First-gate breakdown */}
      {Object.keys(data.blocked_analysis.by_first_gate ?? {}).length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-gray-300">
            First-Gate Breakdown
          </h3>
          <div className="space-y-1">
            {Object.entries(data.blocked_analysis.by_first_gate)
              .slice(0, 8)
              .map(([gate, count]) => {
                const ratio =
                  data.blocked_analysis.total_blocked > 0
                    ? count / data.blocked_analysis.total_blocked
                    : 0;
                return (
                  <div key={gate} className="flex items-center gap-3 text-xs">
                    <span className="w-44 truncate font-mono text-gray-700 dark:text-gray-300">
                      {gate}
                    </span>
                    <span className="w-8 text-right font-semibold">{count}</span>
                    <div className="flex-1 rounded-full bg-gray-200 dark:bg-gray-700 h-2">
                      <div
                        className="h-2 rounded-full bg-blue-500"
                        style={{ width: `${(ratio * 100).toFixed(1)}%` }}
                      />
                    </div>
                    <span className="w-10 text-right text-gray-500">
                      {pct(ratio)}
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {data.total_records === 0 && (
        <p className="text-sm text-gray-400 dark:text-gray-500 text-center py-8">
          No trade decisions recorded in the last {lookback} days.
          <br />
          Diagnostics will populate once trades are processed.
        </p>
      )}
    </div>
  );
}
