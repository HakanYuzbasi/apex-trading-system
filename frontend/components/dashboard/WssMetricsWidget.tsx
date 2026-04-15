"use client";

import { useEffect, useState } from "react";
import { Activity, Shield, Zap, TrendingUp, TrendingDown, RefreshCw, AlertCircle, Clock, LayoutGrid, Wifi, WifiOff } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn, getToneClass } from "@/lib/utils";

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
      <div className="glass-card rounded-xl p-4 flex items-center gap-3 animate-in shake duration-300">
        <AlertCircle className="h-4 w-4 text-negative" />
        <span className="text-[10px] font-black uppercase text-negative">Telemetery Link Defect: {error}</span>
      </div>
    );
  }

  if (!data) return null;

  const hitRateTone = data.hit_rate >= 0.8 ? "positive" : data.hit_rate >= 0.5 ? "warning" : "negative";
  const totalReconnects = data.equity_reconnects + data.crypto_reconnects;

  return (
    <div className="glass-card rounded-xl p-4 space-y-4 animate-in fade-in duration-500">
      <div className="flex items-center justify-between border-b border-border/20 pb-3">
        <div className="flex items-center gap-2">
           <Wifi size={14} className="text-primary animate-pulse" />
           <h3 className="text-[11px] font-black text-foreground uppercase tracking-tight">Data Stream Health</h3>
        </div>
        <Badge variant="outline" className="text-[9px] h-4.5 font-bold bg-background/40 font-mono">
          UP:{uptime(data.session_uptime_seconds)}
        </Badge>
      </div>

      <div className="space-y-2.5">
        {/* Hit rate */}
        <div className="flex items-center justify-between group">
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest group-hover:text-foreground transition-colors">Cache Hit Rate</span>
          <Badge variant={hitRateTone} className="font-mono text-[10px] font-black min-w-[5ch] justify-center">
            {pct(data.hit_rate)}
          </Badge>
        </div>

        {/* Symbols */}
        <div className="flex items-center justify-between group">
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest group-hover:text-foreground transition-colors">Active Subscriptions</span>
          <span className="text-[11px] font-black font-mono text-foreground">{data.cached_symbols} Syms</span>
        </div>

        {/* Equity */}
        <div className="flex items-center justify-between group">
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest group-hover:text-foreground transition-colors">Equity Link</span>
          <div className="flex items-center gap-2">
            {data.equity_reconnects > 0 && (
              <Badge variant="warning" className="text-[9px] h-4 px-1 font-bold">
                {data.equity_reconnects}↺
              </Badge>
            )}
            <Badge variant={data.equity_connected ? "positive" : "negative"} className="text-[9px] h-4.5 px-1.5 font-black uppercase">
              {data.equity_connected ? "Live" : "No_Sig"}
            </Badge>
          </div>
        </div>

        {/* Crypto */}
        <div className="flex items-center justify-between group">
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest group-hover:text-foreground transition-colors">Crypto Link</span>
          <div className="flex items-center gap-2">
            {data.crypto_reconnects > 0 && (
              <Badge variant="warning" className="text-[9px] h-4 px-1 font-bold">
                {data.crypto_reconnects}↺
              </Badge>
            )}
            <Badge variant={data.crypto_connected ? "positive" : "negative"} className="text-[9px] h-4.5 px-1.5 font-black uppercase">
              {data.crypto_connected ? "Live" : "No_Sig"}
            </Badge>
          </div>
        </div>

        {/* REST fallbacks */}
        <div className="flex items-center justify-between group">
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest group-hover:text-foreground transition-colors">REST Fallbacks</span>
          <span className={cn(
            "text-[11px] font-black font-mono tracking-tighter",
            data.wss_misses > data.wss_hits * 0.2 ? "text-warning" : "text-foreground"
          )}>
            {data.wss_misses.toLocaleString()}
          </span>
        </div>
      </div>

      {totalReconnects > 5 && (
        <div className="pt-2 border-t border-border/20 flex items-center gap-2">
          <AlertCircle size={10} className="text-warning animate-pulse" />
          <p className="text-[9px] font-black text-warning uppercase">Stability Alert: {totalReconnects} Sessions Interrupted</p>
        </div>
      )}
    </div>
  );
}
