"use client";

import { useEffect, useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface HeatmapData {
  symbols: string[];
  matrix: (number | null)[][];
  generated_at: string;
  lookback_bars: number;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/** Map correlation [-1, 1] to a Tailwind background class. */
function corrColor(v: number | null): string {
  if (v === null) return "bg-gray-100 dark:bg-gray-800";
  if (v >= 0.8) return "bg-red-600 text-white";
  if (v >= 0.6) return "bg-red-400 text-white";
  if (v >= 0.4) return "bg-red-200 dark:bg-red-900";
  if (v >= 0.2) return "bg-red-50 dark:bg-red-950";
  if (v > -0.2) return "bg-gray-50 dark:bg-gray-800 text-gray-400";
  if (v > -0.4) return "bg-blue-50 dark:bg-blue-950";
  if (v > -0.6) return "bg-blue-200 dark:bg-blue-900";
  if (v > -0.8) return "bg-blue-400 text-white";
  return "bg-blue-600 text-white";
}

function shortSym(sym: string): string {
  return sym.replace("CRYPTO:", "").replace("/USD", "").slice(0, 6);
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function CorrelationHeatmapPanel() {
  const [data, setData] = useState<HeatmapData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [bars, setBars] = useState(60);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/v1/correlation-heatmap?lookback_bars=${bars}`,
        { cache: "no-store" }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load heatmap");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  }, [bars]);

  if (loading && !data)
    return (
      <div className="p-6 text-sm text-gray-500 dark:text-gray-400">
        Loading correlation heatmap…
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

  if (!data || data.symbols.length === 0)
    return (
      <div className="p-6 text-sm text-gray-400 dark:text-gray-500 text-center py-12">
        No symbols with sufficient price history yet.
        <br />
        The heatmap populates after the engine runs for a few minutes.
      </div>
    );

  const n = data.symbols.length;

  return (
    <div className="space-y-4 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Position Correlation Heatmap</h2>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            {n} symbols · {data.lookback_bars}-bar rolling window
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={bars}
            onChange={(e) => setBars(Number(e.target.value))}
            className="rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm"
          >
            {[20, 40, 60, 100, 200].map((b) => (
              <option key={b} value={b}>
                {b} bars
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

      {/* Legend */}
      <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
        <span>−1</span>
        {["bg-blue-600", "bg-blue-400", "bg-blue-200", "bg-gray-100", "bg-red-200", "bg-red-400", "bg-red-600"].map(
          (c) => (
            <span key={c} className={`inline-block h-3 w-5 rounded ${c}`} />
          )
        )}
        <span>+1</span>
      </div>

      {/* Matrix */}
      <div className="overflow-auto">
        <table className="border-collapse text-[10px]">
          <thead>
            <tr>
              <th className="w-14" />
              {data.symbols.map((s) => (
                <th
                  key={s}
                  className="px-0.5 py-1 font-mono font-normal text-gray-500 dark:text-gray-400 whitespace-nowrap"
                  title={s}
                >
                  <span className="block rotate-[-45deg] origin-left ml-2 w-10 truncate">
                    {shortSym(s)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.symbols.map((rowSym, i) => (
              <tr key={rowSym}>
                <td
                  className="pr-2 py-0.5 font-mono text-right text-[10px] text-gray-500 dark:text-gray-400 whitespace-nowrap"
                  title={rowSym}
                >
                  {shortSym(rowSym)}
                </td>
                {data.matrix[i].map((v, j) => (
                  <td
                    key={j}
                    className={`w-7 h-7 text-center align-middle rounded-sm m-px ${corrColor(v)}`}
                    title={`${data.symbols[i]} × ${data.symbols[j]}: ${v?.toFixed(2) ?? "n/a"}`}
                  >
                    {v !== null ? (
                      <span className="text-[9px] font-mono leading-none">
                        {v.toFixed(1)}
                      </span>
                    ) : (
                      <span className="text-gray-300">–</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
