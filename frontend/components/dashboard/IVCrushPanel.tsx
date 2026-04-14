"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";

interface IVSignal {
  symbol: string;
  days_to_earnings?: number | null;
  iv_elevation: number;
  signal: number;
  confidence: number;
  strategy: string;
  earnings_date?: string | null;
  last_updated?: number;
}

interface IVCrushData {
  available: boolean;
  note?: string;
  total_tracked?: number;
  active_signals?: IVSignal[];
  upcoming_earnings?: IVSignal[];
  iv_elevation_threshold?: number;
  iv_crush_scale?: number;
  pead_scale?: number;
}

const STRATEGY_STYLES: Record<string, string> = {
  iv_crush:   "text-orange-400 bg-orange-400/10 border-orange-400/30",
  pead_long:  "text-green-400 bg-green-400/10 border-green-400/30",
  pead_short: "text-red-400 bg-red-400/10 border-red-400/30",
  none:       "text-muted-foreground bg-secondary/20 border-border/30",
};

function StrategyBadge({ strategy }: { strategy: string }) {
  const cls = STRATEGY_STYLES[strategy] ?? STRATEGY_STYLES.none;
  const label = strategy === "iv_crush" ? "IV Crush" : strategy === "pead_long" ? "PEAD ↑" : strategy === "pead_short" ? "PEAD ↓" : "—";
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${cls}`}>
      {label}
    </span>
  );
}

function IVBar({ elevation, threshold }: { elevation: number; threshold: number }) {
  const maxEl = Math.max(threshold * 1.5, 2.0);
  const pct = Math.min((elevation / maxEl) * 100, 100);
  const color = elevation >= threshold ? "bg-orange-400" : elevation >= 1.2 ? "bg-yellow-400" : "bg-muted-foreground/30";
  return (
    <div className="h-1.5 rounded-full bg-secondary/40 overflow-hidden">
      <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function IVCrushPanel() {
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<IVCrushData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"signals" | "upcoming">("signals");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch("/api/v1/iv-crush", {
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

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading IV crush strategy…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "IV crush strategy unavailable — engine not running."}
      </div>
    );
  }

  const threshold = data.iv_elevation_threshold ?? 1.4;
  const signals = data.active_signals ?? [];
  const upcoming = data.upcoming_earnings ?? [];

  return (
    <div className="p-4 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">Earnings IV Crush Strategy</h2>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            {data.total_tracked ?? 0} tracked · IV threshold ×{threshold.toFixed(2)} · {signals.length} active signals
          </p>
        </div>
        <div className="text-right text-[10px] font-mono text-muted-foreground space-y-0.5">
          <div>IV Crush scale: {((data.iv_crush_scale ?? 0.12) * 100).toFixed(0)}%</div>
          <div>PEAD scale: {((data.pead_scale ?? 0.15) * 100).toFixed(0)}%</div>
        </div>
      </div>

      {/* Tab switcher */}
      <div className="flex gap-2">
        {(["signals", "upcoming"] as const).map(tab => (
          <button
            key={tab}
            type="button"
            onClick={() => setActiveTab(tab)}
            className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
              activeTab === tab
                ? "bg-primary text-primary-foreground"
                : "bg-secondary/40 text-muted-foreground hover:bg-secondary/60"
            }`}
          >
            {tab === "signals" ? `Active Signals (${signals.length})` : `Upcoming Earnings (${upcoming.length})`}
          </button>
        ))}
      </div>

      {/* Signal list */}
      {activeTab === "signals" && (
        <>
          {signals.length === 0 ? (
            <div className="flex items-center justify-center rounded-lg border border-border/40 bg-muted/5 p-8">
              <p className="text-sm text-muted-foreground">No active IV signals — signals appear 1-5 days before earnings when IV is elevated.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {signals.map((s, i) => (
                <div key={i} className="rounded-xl border border-border/50 bg-background/50 p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-semibold text-foreground">{s.symbol}</span>
                        <StrategyBadge strategy={s.strategy} />
                      </div>
                      <p className="text-[10px] text-muted-foreground font-mono">
                        {s.earnings_date ?? "—"} · {s.days_to_earnings != null ? `${s.days_to_earnings}d` : "—"}
                      </p>
                    </div>
                    <div className="text-right space-y-0.5">
                      <p className={`text-lg font-bold font-mono ${s.signal < 0 ? "text-orange-400" : "text-green-400"}`}>
                        {s.signal > 0 ? "+" : ""}{(s.signal * 100).toFixed(1)}%
                      </p>
                      <p className="text-[10px] text-muted-foreground font-mono">
                        conf {(s.confidence * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-[10px] text-muted-foreground">
                      <span>IV elevation</span>
                      <span className={`font-mono font-semibold ${s.iv_elevation >= threshold ? "text-orange-400" : "text-muted-foreground"}`}>
                        ×{s.iv_elevation.toFixed(2)}
                      </span>
                    </div>
                    <IVBar elevation={s.iv_elevation} threshold={threshold} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Upcoming earnings list */}
      {activeTab === "upcoming" && (
        <>
          {upcoming.length === 0 ? (
            <div className="flex items-center justify-center rounded-lg border border-border/40 bg-muted/5 p-8">
              <p className="text-sm text-muted-foreground">No upcoming earnings within 7 days for tracked symbols.</p>
            </div>
          ) : (
            <div className="space-y-1.5">
              <div className="grid grid-cols-4 text-[10px] text-muted-foreground px-2 pb-1 border-b border-border/30">
                <span>Symbol</span>
                <span className="text-right">Days</span>
                <span className="text-right">IV Elev</span>
                <span className="text-right">Strategy</span>
              </div>
              {upcoming.map((s, i) => (
                <div key={i} className="grid grid-cols-4 items-center text-[11px] px-2 py-1 rounded hover:bg-secondary/20">
                  <span className="font-mono font-semibold text-foreground">{s.symbol}</span>
                  <span className="text-right font-mono text-foreground">{s.days_to_earnings != null ? `${s.days_to_earnings}d` : "—"}</span>
                  <span className={`text-right font-mono ${s.iv_elevation >= threshold ? "text-orange-400" : "text-muted-foreground"}`}>
                    ×{s.iv_elevation.toFixed(2)}
                  </span>
                  <div className="flex justify-end">
                    <StrategyBadge strategy={s.strategy} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-[10px] text-muted-foreground pt-1 border-t border-border/30">
        <span>IV Crush = sell vol pre-earnings</span>
        <span>PEAD = follow post-earnings price gap</span>
        <span>IV elev ≥ ×{threshold.toFixed(1)} triggers signal</span>
      </div>
    </div>
  );
}
