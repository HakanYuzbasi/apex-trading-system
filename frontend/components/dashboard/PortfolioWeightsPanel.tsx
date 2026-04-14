"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";

interface PortfolioWeightsData {
  available: boolean;
  note?: string;
  method?: string;
  n_symbols?: number;
  computed_at?: number;
  cov_condition?: number;
  weights?: Record<string, number>;
  top_signals?: Record<string, number>;
}

function WeightBar({ symbol, weight, signal }: { symbol: string; weight: number; signal?: number }) {
  const pct = Math.min(Math.max(weight, 0), 1) * 100;
  const barColor = weight >= 0.12 ? "bg-primary" : weight >= 0.06 ? "bg-blue-400" : "bg-muted-foreground/40";
  const sigColor =
    signal == null ? "" : signal >= 0.18 ? "text-green-400" : signal >= 0.10 ? "text-blue-400" : "text-muted-foreground";

  return (
    <div className="flex items-center gap-2">
      <span className="w-20 shrink-0 truncate font-mono text-[11px] text-foreground font-semibold" title={symbol}>
        {symbol.replace("CRYPTO:", "").replace("/USD", "")}
      </span>
      <div className="flex-1 h-2 rounded-full bg-secondary/40 overflow-hidden">
        <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${(pct / 15 * 100).toFixed(1)}%` }} />
      </div>
      <span className="w-10 text-right font-mono text-[11px] text-foreground">{(weight * 100).toFixed(1)}%</span>
      {signal != null && (
        <span className={`w-12 text-right font-mono text-[10px] ${sigColor}`}>
          sig {(signal * 100).toFixed(0)}
        </span>
      )}
    </div>
  );
}

function fmtTs(epoch?: number): string {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toISOString().slice(11, 19) + " UTC";
}

export default function PortfolioWeightsPanel() {
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<PortfolioWeightsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch("/api/v1/portfolio-weights", {
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
    const id = setInterval(load, 60_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [token]);

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading portfolio weights…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "Portfolio weights unavailable — engine may not be running or weights not yet computed."}
      </div>
    );
  }

  const weights = Object.entries(data.weights ?? {}).sort((a, b) => b[1] - a[1]);
  const signals = data.top_signals ?? {};
  const totalWeight = weights.reduce((s, [, w]) => s + w, 0);

  return (
    <div className="p-4 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">Portfolio Target Weights</h2>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            Signal-aware HRP — {data.method ?? "—"} · {data.n_symbols ?? 0} symbols
          </p>
        </div>
        <div className="text-right">
          <p className="text-[11px] text-muted-foreground font-mono">{fmtTs(data.computed_at)}</p>
          {data.cov_condition != null && (
            <p className={`text-[10px] font-mono mt-0.5 ${data.cov_condition > 1000 ? "text-yellow-400" : "text-muted-foreground"}`}>
              cov κ={data.cov_condition.toFixed(0)}
            </p>
          )}
        </div>
      </div>

      {/* Summary chips */}
      <div className="flex flex-wrap gap-2 text-[11px]">
        <span className="rounded-lg border border-border/50 bg-background/50 px-3 py-1.5 text-muted-foreground font-mono">
          {weights.length} symbols
        </span>
        <span className="rounded-lg border border-border/50 bg-background/50 px-3 py-1.5 text-muted-foreground font-mono">
          total {(totalWeight * 100).toFixed(1)}%
        </span>
        {weights.length > 0 && (
          <span className="rounded-lg border border-border/50 bg-background/50 px-3 py-1.5 text-muted-foreground font-mono">
            max {(weights[0][1] * 100).toFixed(1)}%
          </span>
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-[10px] text-muted-foreground">
        <span>Bar = weight (max 15%)</span>
        <span>Sig = ML signal strength</span>
      </div>

      {/* Weight bars */}
      {weights.length === 0 ? (
        <div className="flex items-center justify-center rounded-lg border border-border/40 bg-muted/5 p-8">
          <p className="text-sm text-muted-foreground">No weights computed yet — engine needs at least 10 bars per symbol.</p>
        </div>
      ) : (
        <div className="space-y-1.5">
          {weights.map(([sym, w]) => (
            <WeightBar
              key={sym}
              symbol={sym}
              weight={w}
              signal={signals[sym]}
            />
          ))}
        </div>
      )}
    </div>
  );
}
