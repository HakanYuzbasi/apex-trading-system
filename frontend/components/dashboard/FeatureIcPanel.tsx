"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";

interface FeatureRow {
  feature: string;
  ic_30d: number;
  ic_90d: number;
  n_obs: number;
  status: "strong" | "live" | "suspect" | "dead";
}

interface IcData {
  available: boolean;
  note?: string;
  n_features?: number;
  pending_snapshots?: number;
  dead_count?: number;
  strong_count?: number;
  dead?: string[];
  strong?: string[];
  features?: FeatureRow[];
  thresholds?: { dead: number; suspect: number; strong: number };
}

const STATUS_STYLES: Record<string, string> = {
  strong: "text-green-400 bg-green-400/10 border-green-400/30",
  live: "text-blue-400 bg-blue-400/10 border-blue-400/30",
  suspect: "text-yellow-400 bg-yellow-400/10 border-yellow-400/30",
  dead: "text-red-400 bg-red-400/10 border-red-400/30",
};

function IcBar({ value, max = 0.12 }: { value: number; max?: number }) {
  const abs = Math.min(Math.abs(value), max);
  const pct = (abs / max) * 100;
  const color = value > 0.05 ? "bg-green-400" : value < 0 ? "bg-red-400" : value > 0.015 ? "bg-blue-400" : "bg-red-300";
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-1.5 rounded-full bg-secondary/40 overflow-hidden">
        <div
          className={`h-full rounded-full ${color} transition-all`}
          style={{ width: `${pct.toFixed(1)}%` }}
        />
      </div>
      <span className={`text-[11px] font-mono w-14 text-right ${value >= 0.05 ? "text-green-400" : value < 0.015 ? "text-red-400" : "text-muted-foreground"}`}>
        {value >= 0 ? "+" : ""}{(value * 100).toFixed(2)}%
      </span>
    </div>
  );
}

function SummaryChip({ label, value, color }: { label: string; value: number | string; color?: string }) {
  return (
    <div className="flex flex-col items-center rounded-lg border border-border/50 bg-background/50 px-4 py-2.5 gap-0.5">
      <span className="text-[10px] text-muted-foreground uppercase tracking-wide">{label}</span>
      <span className={`text-xl font-bold font-mono ${color ?? "text-foreground"}`}>{value}</span>
    </div>
  );
}

export default function FeatureIcPanel() {
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<IcData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch("/api/v1/feature-ic", {
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
    const id = setInterval(load, 120_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [token]);

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading IC data…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "Feature IC tracking not available — engine may not be running."}
      </div>
    );
  }

  const allFeatures = data.features ?? [];
  const rows = filter === "all" ? allFeatures : allFeatures.filter((f) => f.status === filter);

  return (
    <div className="p-4 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">Feature IC Tracker</h2>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            Spearman rank correlation between feature values and 5-day forward returns
          </p>
        </div>
        <span className="text-[11px] text-muted-foreground font-mono">
          {data.pending_snapshots ?? 0} pending
        </span>
      </div>

      {/* Summary row */}
      <div className="flex flex-wrap gap-3">
        <SummaryChip label="Features" value={data.n_features ?? 0} />
        <SummaryChip label="Strong (IC>5%)" value={data.strong_count ?? 0} color="text-green-400" />
        <SummaryChip label="Dead (IC<1.5%)" value={data.dead_count ?? 0} color="text-red-400" />
      </div>

      {/* Filter chips */}
      <div className="flex flex-wrap gap-2">
        {(["all", "strong", "live", "suspect", "dead"] as const).map((s) => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={`px-3 py-1 rounded-lg border text-xs transition-colors
              ${filter === s ? "border-primary/60 bg-primary/10 text-primary" : "border-border/50 bg-background/50 text-muted-foreground hover:border-border"}`}
          >
            {s.charAt(0).toUpperCase() + s.slice(1)}
            {s !== "all" && (
              <span className="ml-1.5 font-mono font-semibold">
                {allFeatures.filter((f) => f.status === s).length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Thresholds legend */}
      {data.thresholds && (
        <div className="flex gap-4 text-[11px] text-muted-foreground">
          <span>
            <span className="text-red-400 font-semibold">Dead</span> {`< ${(data.thresholds.dead * 100).toFixed(1)}%`}
          </span>
          <span>
            <span className="text-yellow-400 font-semibold">Suspect</span> {`< ${(data.thresholds.suspect * 100).toFixed(1)}%`}
          </span>
          <span>
            <span className="text-green-400 font-semibold">Strong</span> {`> ${(data.thresholds.strong * 100).toFixed(1)}%`}
          </span>
        </div>
      )}

      {rows.length === 0 && (
        <div className="flex items-center justify-center rounded-lg border border-border/40 bg-secondary/10 p-8">
          <p className="text-sm text-muted-foreground">No features match the selected filter.</p>
        </div>
      )}

      {/* Feature table */}
      {rows.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-border/60">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-border/60 text-muted-foreground text-[11px] uppercase">
                <th className="px-3 py-2 text-left w-48">Feature</th>
                <th className="px-3 py-2 text-left">30d IC</th>
                <th className="px-3 py-2 text-right">90d IC</th>
                <th className="px-3 py-2 text-right">Obs</th>
                <th className="px-3 py-2 text-left">Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.feature} className="border-b border-border/40 hover:bg-secondary/30 transition-colors">
                  <td className="px-3 py-2 text-foreground truncate max-w-[180px]" title={r.feature}>
                    {r.feature}
                  </td>
                  <td className="px-3 py-2 w-48">
                    <IcBar value={r.ic_30d} />
                  </td>
                  <td className={`px-3 py-2 text-right ${r.ic_90d >= 0.05 ? "text-green-400" : r.ic_90d < 0.015 ? "text-red-400" : "text-muted-foreground"}`}>
                    {r.ic_90d >= 0 ? "+" : ""}{(r.ic_90d * 100).toFixed(2)}%
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">{r.n_obs}</td>
                  <td className="px-3 py-2">
                    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${STATUS_STYLES[r.status] ?? ""}`}>
                      {r.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
