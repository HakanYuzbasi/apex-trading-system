"use client";

import { useEffect, useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface WssMetrics {
  hit_rate: number;
  wss_hits: number;
  wss_misses: number;
  equity_reconnects: number;
  crypto_reconnects: number;
  equity_connected: boolean;
  crypto_connected: boolean;
  session_uptime_seconds: number;
  cached_symbols: number;
  error?: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function pct(v: number): string {
  return (v * 100).toFixed(1) + "%";
}

function uptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function WssMetricsWidget() {
  const [data, setData] = useState<WssMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const res = await fetch("/api/v1/wss-metrics", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (json.error) {
        setError(json.error);
        setData(null);
      } else {
        setData(json);
        setError(null);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed");
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 10_000);
    return () => clearInterval(id);
  }, []);

  if (error) {
    return (
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 text-xs text-gray-400">
        WSS Metrics: {error}
      </div>
    );
  }

  if (!data) return null;

  const hitRateColor =
    data.hit_rate >= 0.8
      ? "text-green-600 dark:text-green-400"
      : data.hit_rate >= 0.5
      ? "text-amber-600 dark:text-amber-400"
      : "text-red-500";

  const totalReconnects = data.equity_reconnects + data.crypto_reconnects;

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 text-xs">
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold text-gray-700 dark:text-gray-300">WSS Health</span>
        <span className="text-gray-400 text-[10px]">
          ↑ {uptime(data.session_uptime_seconds)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        {/* Hit rate */}
        <span className="text-gray-500">Cache Hit Rate</span>
        <span className={`font-semibold text-right ${hitRateColor}`}>
          {pct(data.hit_rate)}
        </span>

        {/* Symbols */}
        <span className="text-gray-500">Cached Symbols</span>
        <span className="text-right">{data.cached_symbols}</span>

        {/* Equity */}
        <span className="text-gray-500">Equity</span>
        <span className="text-right flex items-center justify-end gap-1">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              data.equity_connected ? "bg-green-500" : "bg-red-400"
            }`}
          />
          {data.equity_connected ? "live" : "offline"}
          {data.equity_reconnects > 0 && (
            <span className="text-amber-500 ml-1">
              ({data.equity_reconnects}↺)
            </span>
          )}
        </span>

        {/* Crypto */}
        <span className="text-gray-500">Crypto</span>
        <span className="text-right flex items-center justify-end gap-1">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              data.crypto_connected ? "bg-green-500" : "bg-red-400"
            }`}
          />
          {data.crypto_connected ? "live" : "offline"}
          {data.crypto_reconnects > 0 && (
            <span className="text-amber-500 ml-1">
              ({data.crypto_reconnects}↺)
            </span>
          )}
        </span>

        {/* REST fallbacks */}
        <span className="text-gray-500">REST Fallbacks</span>
        <span
          className={`text-right ${
            data.wss_misses > data.wss_hits * 0.2 ? "text-amber-500" : ""
          }`}
        >
          {data.wss_misses.toLocaleString()}
        </span>
      </div>

      {totalReconnects > 5 && (
        <div className="mt-2 text-amber-600 dark:text-amber-400 text-[10px]">
          ⚠ {totalReconnects} reconnects this session
        </div>
      )}
    </div>
  );
}
