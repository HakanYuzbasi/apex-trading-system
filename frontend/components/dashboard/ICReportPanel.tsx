"use client";

import { useEffect, useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface ICTrackerData {
  summary: Record<string, number>;
  dead_features: string[];
  strong_features: string[];
  pending_count: number;
  observation_counts: Record<string, number>;
  error?: string;
}

interface BucketStats {
  label: string;
  n_obs: number;
  ic: number;
  hit_rate: number;
  mean_return: number;
}

interface RegimeDecay {
  optimal_hold_hours: number;
  alpha_half_life: number | null;
  peak_ic: number;
  buckets: BucketStats[];
}

interface AlphaDecayData {
  total_trades: number;
  updated_at: string;
  regimes: Record<string, RegimeDecay>;
  error?: string;
}

interface ICReport {
  ic_tracker?: ICTrackerData;
  alpha_decay?: AlphaDecayData;
  error?: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function icColor(ic: number): string {
  if (ic >= 0.05) return "text-green-600 dark:text-green-400";
  if (ic >= 0.03) return "text-amber-600 dark:text-amber-400";
  if (ic >= 0.015) return "text-gray-500 dark:text-gray-400";
  return "text-red-500 dark:text-red-400";
}

function statusBadge(feature: string, dead: string[], strong: string[]) {
  if (strong.includes(feature))
    return (
      <span className="ml-1 px-1.5 py-0.5 rounded text-[9px] bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300">
        STRONG
      </span>
    );
  if (dead.includes(feature))
    return (
      <span className="ml-1 px-1.5 py-0.5 rounded text-[9px] bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300">
        DEAD
      </span>
    );
  return null;
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function ICReportPanel() {
  const [data, setData] = useState<ICReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/ic-report", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 60_000); // refresh every 60s
    return () => clearInterval(id);
  }, []);

  if (loading && !data)
    return (
      <div className="p-6 text-sm text-gray-500 dark:text-gray-400">
        Loading IC report…
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

  const ict = data?.ic_tracker;
  const adc = data?.alpha_decay;

  return (
    <div className="space-y-6 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Information Coefficient Report</h2>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            Rolling Spearman IC per signal component · Alpha decay by hold horizon
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="text-xs text-blue-500 hover:underline disabled:opacity-50"
        >
          {loading ? "refreshing…" : "refresh"}
        </button>
      </div>

      {/* IC Tracker Section */}
      {ict && !ict.error ? (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              Feature IC Summary
            </h3>
            <span className="text-xs text-gray-400">
              {ict.pending_count} pending fills
            </span>
          </div>

          {/* Dead / Strong callouts */}
          {ict.dead_features.length > 0 && (
            <div className="rounded border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950 px-3 py-2 text-xs text-red-700 dark:text-red-300">
              ⚠ Dead features (IC &lt; 0.015): {ict.dead_features.join(", ")}
              <span className="ml-2 text-red-500">— signal dampened 20%</span>
            </div>
          )}
          {ict.strong_features.length > 0 && (
            <div className="rounded border border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-950 px-3 py-2 text-xs text-green-700 dark:text-green-300">
              ✓ Strong features (IC &gt; 0.05): {ict.strong_features.join(", ")}
              <span className="ml-2 text-green-500">— confidence boosted 8%</span>
            </div>
          )}

          {/* Feature table */}
          <div className="overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400">
                  <th className="text-left py-1 pr-4 font-medium">Feature</th>
                  <th className="text-right py-1 pr-4 font-medium">IC (30d)</th>
                  <th className="text-right py-1 pr-4 font-medium">Obs</th>
                  <th className="text-left py-1 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(ict.summary).map(([feat, ic]) => (
                  <tr
                    key={feat}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <td className="py-1 pr-4 font-mono text-gray-700 dark:text-gray-300">
                      {feat}
                    </td>
                    <td className={`py-1 pr-4 text-right font-semibold font-mono ${icColor(ic)}`}>
                      {ic.toFixed(4)}
                    </td>
                    <td className="py-1 pr-4 text-right text-gray-500">
                      {ict.observation_counts[feat] ?? "—"}
                    </td>
                    <td className="py-1">
                      {statusBadge(feat, ict.dead_features, ict.strong_features)}
                    </td>
                  </tr>
                ))}
                {Object.keys(ict.summary).length === 0 && (
                  <tr>
                    <td colSpan={4} className="py-4 text-center text-gray-400">
                      No features with ≥10 observations yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        ict?.error && (
          <div className="text-xs text-red-400">IC Tracker: {ict.error}</div>
        )
      )}

      {/* Alpha Decay Section */}
      {adc && !adc.error ? (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              Alpha Decay by Hold Horizon
            </h3>
            <span className="text-xs text-gray-400">
              {adc.total_trades} total trades
            </span>
          </div>
          {Object.entries(adc.regimes).length === 0 ? (
            <p className="text-xs text-gray-400 py-4 text-center">
              No regime data yet — needs completed trades per regime.
            </p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {Object.entries(adc.regimes).map(([regime, rd]) => (
                <div
                  key={regime}
                  className="rounded-lg border border-gray-200 dark:border-gray-700 p-3"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-semibold capitalize text-gray-700 dark:text-gray-300">
                      {regime}
                    </span>
                    <span className="text-[10px] text-gray-400">
                      opt {rd.optimal_hold_hours}h
                      {rd.alpha_half_life != null &&
                        ` · half-life ${rd.alpha_half_life}h`}
                    </span>
                  </div>
                  <div className="grid grid-cols-4 gap-1">
                    {rd.buckets.map((b) => (
                      <div key={b.label} className="text-center">
                        <div
                          className={`text-[10px] font-mono font-semibold ${icColor(b.ic)}`}
                        >
                          {b.n_obs >= 10 ? b.ic.toFixed(3) : "—"}
                        </div>
                        <div className="text-[9px] text-gray-400">{b.label}</div>
                        <div className="text-[9px] text-gray-400">n={b.n_obs}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        adc?.error && (
          <div className="text-xs text-red-400">Alpha Decay: {adc.error}</div>
        )
      )}
    </div>
  );
}
