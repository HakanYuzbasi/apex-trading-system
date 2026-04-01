"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/lib/auth-context";

interface WinRates {
  paper: number;
  live: number;
  n: number;
}

interface PaperTrade {
  symbol: string;
  side: string;
  entry_price: number;
  exit_price: number;
  notional: number;
  pnl_usd: number;
  live_pnl_usd: number;
  shortfall_usd: number;
  entry_ts: number;
  exit_ts: number;
}

interface PaperAccountData {
  available: boolean;
  note?: string;
  open_positions?: number;
  closed_trades?: number;
  paper_total_pnl?: number;
  live_total_pnl?: number;
  implementation_shortfall_usd?: number;
  shortfall_pct?: number;
  avg_shortfall_per_trade?: number;
  win_rates?: WinRates;
  day_start_ts?: number;
  recent_trades?: PaperTrade[];
}

function fmtUsd(v?: number): string {
  if (v == null) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}$${Math.abs(v).toFixed(2)}`;
}

function fmtTs(epoch?: number): string {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toISOString().slice(11, 19) + " UTC";
}

function ShortfallGauge({ pct }: { pct: number }) {
  const abs = Math.min(Math.abs(pct), 100);
  const color = abs < 5 ? "bg-green-400" : abs < 15 ? "bg-yellow-400" : "bg-red-400";
  return (
    <div className="space-y-1">
      <div className="h-2 rounded-full bg-secondary/40 overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${abs}%` }} />
      </div>
      <p className={`text-[10px] font-mono text-right ${abs < 5 ? "text-green-400" : abs < 15 ? "text-yellow-400" : "text-red-400"}`}>
        {pct.toFixed(1)}% of paper P&L
      </p>
    </div>
  );
}

export default function PaperAccountPanel() {
  const { token } = useAuthContext();
  const [data, setData] = useState<PaperAccountData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch("/api/v1/paper-account", {
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

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading paper account…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "Paper account unavailable — engine not running."}
      </div>
    );
  }

  const shortfall = data.implementation_shortfall_usd ?? 0;
  const shortfallPct = data.shortfall_pct ?? 0;
  const wr = data.win_rates ?? { paper: 0, live: 0, n: 0 };
  const trades = data.recent_trades ?? [];

  return (
    <div className="p-4 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">Shadow Paper Account</h2>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            Implementation shortfall tracker · {data.closed_trades ?? 0} closed trades
          </p>
        </div>
        <span className="text-[10px] text-muted-foreground font-mono">
          since {fmtTs(data.day_start_ts)}
        </span>
      </div>

      {/* P&L comparison */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl border border-border/60 bg-background/50 p-3 space-y-1">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Paper P&L (theory)</p>
          <p className={`text-xl font-bold font-mono ${(data.paper_total_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
            {fmtUsd(data.paper_total_pnl)}
          </p>
          <p className="text-[10px] text-muted-foreground">At mid prices, no slippage</p>
        </div>
        <div className="rounded-xl border border-border/60 bg-background/50 p-3 space-y-1">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Live P&L (actual)</p>
          <p className={`text-xl font-bold font-mono ${(data.live_total_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
            {fmtUsd(data.live_total_pnl)}
          </p>
          <p className="text-[10px] text-muted-foreground">After slippage + commissions</p>
        </div>
      </div>

      {/* Implementation shortfall */}
      <div className="rounded-xl border border-border/60 bg-background/50 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Implementation Shortfall</p>
            <p className={`text-2xl font-bold font-mono mt-1 ${shortfall <= 0 ? "text-green-400" : shortfall < 50 ? "text-yellow-400" : "text-red-400"}`}>
              {fmtUsd(shortfall)}
            </p>
          </div>
          <div className="text-right space-y-1">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg per Trade</p>
            <p className="text-lg font-mono font-semibold text-foreground">
              {fmtUsd(data.avg_shortfall_per_trade)}
            </p>
          </div>
        </div>
        <ShortfallGauge pct={shortfallPct} />
      </div>

      {/* Win rate comparison */}
      {wr.n > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
            Win Rate Comparison ({wr.n} matched trades)
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-border/50 bg-background/40 p-3 text-center">
              <p className="text-[10px] text-muted-foreground uppercase mb-1">Paper</p>
              <p className="text-lg font-bold font-mono text-blue-400">{(wr.paper * 100).toFixed(0)}%</p>
            </div>
            <div className="rounded-lg border border-border/50 bg-background/40 p-3 text-center">
              <p className="text-[10px] text-muted-foreground uppercase mb-1">Live</p>
              <p className={`text-lg font-bold font-mono ${wr.live >= wr.paper - 0.05 ? "text-green-400" : "text-yellow-400"}`}>
                {(wr.live * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        </section>
      )}

      {/* Recent trades */}
      {trades.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Recent Trades
          </h3>
          <div className="space-y-1">
            <div className="grid grid-cols-5 text-[10px] text-muted-foreground px-1 pb-1 border-b border-border/30">
              <span>Symbol</span>
              <span className="text-right">Paper</span>
              <span className="text-right">Live</span>
              <span className="text-right">Gap</span>
              <span className="text-right">Exit</span>
            </div>
            {trades.slice(0, 10).map((t, i) => {
              const gap = t.shortfall_usd;
              return (
                <div key={i} className="grid grid-cols-5 text-[11px] px-1 py-0.5 rounded hover:bg-secondary/20">
                  <span className="font-mono font-semibold text-foreground truncate">
                    {t.symbol.replace("CRYPTO:", "").replace("/USD", "")}
                  </span>
                  <span className={`text-right font-mono ${t.pnl_usd >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {fmtUsd(t.pnl_usd)}
                  </span>
                  <span className={`text-right font-mono ${t.live_pnl_usd >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {t.live_pnl_usd === 0 ? "—" : fmtUsd(t.live_pnl_usd)}
                  </span>
                  <span className={`text-right font-mono ${gap <= 0 ? "text-green-400" : gap < 10 ? "text-yellow-400" : "text-red-400"}`}>
                    {t.live_pnl_usd === 0 ? "—" : fmtUsd(gap)}
                  </span>
                  <span className="text-right font-mono text-muted-foreground">{fmtTs(t.exit_ts)}</span>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {trades.length === 0 && (
        <div className="flex items-center justify-center rounded-lg border border-border/40 bg-muted/5 p-8">
          <p className="text-sm text-muted-foreground">No closed trades yet — paper account mirrors live entries/exits.</p>
        </div>
      )}
    </div>
  );
}
