"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/lib/auth-context";

interface PairRecord {
  leg_y: string;
  leg_x: string;
  hedge_ratio: number;
  half_life: number;
  z_score: number;
  z_entry: number;
  z_exit: number;
  spread_mean: number;
  spread_std: number;
  last_spread: number;
  corr: number;
  signal_y: number;
  signal_x: number;
  last_updated?: number;
}

interface CrossAssetPairsData {
  available: boolean;
  note?: string;
  n_pairs?: number;
  last_scan_ts?: number;
  active_pairs?: PairRecord[];
  z_entry?: number;
  z_exit?: number;
}

function fmtTs(epoch?: number | null): string {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toISOString().slice(11, 19) + " UTC";
}

function ZScoreBar({ z, zEntry }: { z: number; zEntry: number }) {
  const maxZ = Math.max(zEntry * 2, 4);
  const width = Math.min(Math.abs(z) / maxZ * 100, 100);
  const color = Math.abs(z) >= zEntry
    ? (z > 0 ? "bg-red-400" : "bg-green-400")
    : "bg-muted-foreground/30";
  return (
    <div className="h-1.5 rounded-full bg-secondary/40 overflow-hidden">
      <div
        className={`h-full rounded-full transition-all ${color}`}
        style={{ width: `${width}%`, marginLeft: z < 0 ? `${50 - width / 2}%` : "50%" }}
      />
    </div>
  );
}

function SignalBadge({ value }: { value: number }) {
  if (Math.abs(value) < 0.01) return <span className="text-muted-foreground">—</span>;
  const color = value > 0 ? "text-green-400" : "text-red-400";
  const sign = value > 0 ? "+" : "";
  return <span className={`font-mono font-semibold ${color}`}>{sign}{(value * 100).toFixed(1)}%</span>;
}

export default function CrossAssetPairsPanel() {
  const { token } = useAuthContext();
  const [data, setData] = useState<CrossAssetPairsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch("/api/v1/cross-asset-pairs", {
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
    const id = setInterval(load, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [token]);

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading cross-asset pairs…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "Cross-asset pairs unavailable — engine not running or no pairs found yet."}
      </div>
    );
  }

  const pairs = data.active_pairs ?? [];
  const zEntry = data.z_entry ?? 1.8;

  return (
    <div className="p-4 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">Cross-Asset Pairs Arbitrage</h2>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            {data.n_pairs ?? 0} active pairs · z-entry {zEntry.toFixed(1)}σ · last scan {fmtTs(data.last_scan_ts)}
          </p>
        </div>
        <div className="text-right text-[10px] font-mono text-muted-foreground space-y-0.5">
          <div>entry ≥ {zEntry.toFixed(1)}σ</div>
          <div>exit ≤ {(data.z_exit ?? 0.4).toFixed(1)}σ</div>
        </div>
      </div>

      {/* Pairs table */}
      {pairs.length === 0 ? (
        <div className="flex items-center justify-center rounded-lg border border-border/40 bg-muted/5 p-8">
          <p className="text-sm text-muted-foreground">No pairs discovered yet — first scan runs at startup, then every 2 hours.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {/* Header row */}
          <div className="grid grid-cols-6 text-[10px] text-muted-foreground px-2 pb-1 border-b border-border/30">
            <span className="col-span-2">Pair (Y / X)</span>
            <span className="text-right">Z-score</span>
            <span className="text-right">HL (d)</span>
            <span className="text-right">Sig Y</span>
            <span className="text-right">Sig X</span>
          </div>

          {pairs.map((p, i) => {
            const active = Math.abs(p.z_score) >= zEntry;
            return (
              <div
                key={i}
                className={`rounded-lg border px-2 py-2 space-y-1.5 transition
                  ${active ? "border-primary/30 bg-primary/5" : "border-border/30 bg-background/30"}`}
              >
                {/* Main row */}
                <div className="grid grid-cols-6 items-center text-[11px]">
                  <div className="col-span-2 space-y-0.5">
                    <p className="font-mono font-semibold text-foreground truncate">
                      {p.leg_y.replace("CRYPTO:", "").replace("/USD", "")}
                    </p>
                    <p className="font-mono text-muted-foreground truncate text-[10px]">
                      vs {p.leg_x.replace("CRYPTO:", "").replace("/USD", "")}
                    </p>
                  </div>
                  <div className="text-right space-y-0.5">
                    <p className={`font-mono font-bold text-sm ${active ? (p.z_score > 0 ? "text-red-400" : "text-green-400") : "text-foreground"}`}>
                      {p.z_score.toFixed(2)}σ
                    </p>
                    <p className="text-[10px] text-muted-foreground font-mono">r={p.corr.toFixed(2)}</p>
                  </div>
                  <div className="text-right font-mono text-foreground">
                    {p.half_life.toFixed(1)}d
                  </div>
                  <div className="text-right">
                    <SignalBadge value={p.signal_y} />
                  </div>
                  <div className="text-right">
                    <SignalBadge value={p.signal_x} />
                  </div>
                </div>
                {/* Z-bar */}
                <ZScoreBar z={p.z_score} zEntry={zEntry} />
              </div>
            );
          })}
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-4 text-[10px] text-muted-foreground pt-1">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-400 inline-block" /> below mean (buy Y)</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-400 inline-block" /> above mean (sell Y)</span>
        <span>HL = mean-reversion half-life · Sig = overlay signal strength</span>
      </div>
    </div>
  );
}
