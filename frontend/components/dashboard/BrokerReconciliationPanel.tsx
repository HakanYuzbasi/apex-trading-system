"use client";

import { useMemo } from "react";
import { AlertTriangle, CheckCircle2, RefreshCw, Layers, ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn, getToneClass } from "@/lib/utils";

export interface BrokerPosition {
  symbol: string;
  normalized_symbol: string;
  qty: number;
  side: string;
  market_value: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  current_price: number;
  avg_price: number;
  is_orphaned: boolean;
}

interface Props {
  brokerPositions?: BrokerPosition[];
  lastUpdated?: string | null;
}

function fmt(n: number, decimals = 2): string {
  return Number.isFinite(n) ? n.toFixed(decimals) : "—";
}

function fmtMoney(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return (n >= 0 ? "+" : "") + "$" + Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtValue(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export default function BrokerReconciliationPanel({ brokerPositions, lastUpdated }: Props) {
  const positions = useMemo<BrokerPosition[]>(() => {
    if (!brokerPositions || !Array.isArray(brokerPositions)) return [];
    return [...brokerPositions].sort((a, b) => {
      // Orphaned positions float to the top
      if (a.is_orphaned !== b.is_orphaned) return a.is_orphaned ? -1 : 1;
      return Math.abs(b.market_value) - Math.abs(a.market_value);
    });
  }, [brokerPositions]);

  const orphanCount = useMemo(() => positions.filter((p) => p.is_orphaned).length, [positions]);
  const trackedCount = positions.length - orphanCount;
  const totalOrphanMV = useMemo(
    () => positions.filter((p) => p.is_orphaned).reduce((s, p) => s + p.market_value, 0),
    [positions],
  );

  const isEmpty = positions.length === 0;
  const isStale = !lastUpdated;

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500">
      {/* Header summary */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <div className={cn("flex items-center gap-2.5 px-4 py-2 rounded-xl border backdrop-blur-md transition-all", orphanCount > 0 ? "border-negative/30 bg-negative/5" : "border-border/30 bg-background/20")}>
            <ShieldAlert className={cn("h-4 w-4", orphanCount > 0 ? "text-negative" : "text-muted-foreground")} />
            <span className={cn("text-sm font-bold", orphanCount > 0 ? "text-negative" : "text-muted-foreground")}>{orphanCount} Orphaned</span>
            {orphanCount > 0 && (
              <Badge variant="negative" className="text-[10px] font-mono h-5 bg-negative/20">{fmtValue(totalOrphanMV)}</Badge>
            )}
          </div>
          <div className="flex items-center gap-2.5 px-4 py-2 rounded-xl border border-positive/30 bg-positive/5 backdrop-blur-md">
            <CheckCircle2 className="h-4 w-4 text-positive" />
            <span className="text-sm font-bold text-positive">{trackedCount} Strategy Linked</span>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {isStale ? (
            <div className="flex items-center gap-2 text-[11px] font-bold text-muted-foreground uppercase tracking-widest bg-muted/20 px-3 py-1.5 rounded-full">
              <RefreshCw className="h-3 w-3 animate-spin" />
              Awaiting Sync
            </div>
          ) : (
            <div className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest bg-background/40 border border-border/40 px-3 py-1.5 rounded-full">
               Last: {String(lastUpdated).split("T")[1]?.replace(/\.\d+.*$/, "") ?? lastUpdated}
            </div>
          )}
        </div>
      </div>

      {isEmpty ? (
        <div className="glass-card rounded-2xl border-dashed border-2 py-16 flex flex-col items-center gap-4 text-muted-foreground">
           <Layers className="h-8 w-8 opacity-20" />
           <p className="text-sm font-medium">
             {isStale
               ? "Initializing broker position stream..."
               : "No active exposure at broker. Portfolio is flat."}
           </p>
        </div>
      ) : (
        <div className="glass-card rounded-2xl border border-border/40 overflow-hidden shadow-2xl shadow-black/20">
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-[11px]">
              <thead className="bg-background/60 backdrop-blur-md text-muted-foreground border-b border-border/40">
                <tr>
                  <th className="px-4 py-4 text-left font-bold uppercase tracking-tighter w-24">Status</th>
                  <th className="px-4 py-4 text-left font-bold uppercase tracking-tighter w-24">Symbol</th>
                  <th className="px-4 py-4 text-center font-bold uppercase tracking-tighter w-20">Side</th>
                  <th className="px-4 py-4 text-right font-bold uppercase tracking-tighter">Quantity</th>
                  <th className="px-4 py-4 text-right font-bold uppercase tracking-tighter">Avg Entry</th>
                  <th className="px-4 py-4 text-right font-bold uppercase tracking-tighter">Current</th>
                  <th className="px-4 py-4 text-right font-bold uppercase tracking-tighter">Mkt Value</th>
                  <th className="px-4 py-4 text-right font-bold uppercase tracking-tighter">Unrlzd P&L</th>
                  <th className="px-4 py-4 text-right font-bold uppercase tracking-tighter pr-6">Return</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/20">
                {positions.map((pos, idx) => {
                  const isOrphaned = pos.is_orphaned;
                  const plPositive = pos.unrealized_pl >= 0;
                  
                  // Side normalization logic
                  const rawSide = String(pos.side ?? "").toUpperCase();
                  const cleanSide = rawSide.replace(/^POSITIONSIDE\./i, "").replace(/^POSITION_SIDE\./i, "").trim() || "N/A";
                  const isLong = cleanSide === "LONG" || cleanSide === "BUY";

                  return (
                    <tr
                      key={`${pos.symbol}-${idx}`}
                      className={cn(
                        "transition-all duration-200",
                        isOrphaned
                          ? "bg-negative/[0.03] hover:bg-negative/[0.06]"
                          : "bg-background/20 hover:bg-primary/[0.03]"
                      )}
                    >
                      {/* Status badge */}
                      <td className="px-4 py-4">
                        <Badge 
                          variant={isOrphaned ? "negative" : "positive"} 
                          className="text-[9px] h-4.5 px-2 font-black tracking-tight"
                        >
                          {isOrphaned ? "ORPHAN" : "TRACKED"}
                        </Badge>
                      </td>

                      {/* Symbol */}
                      <td className="px-4 py-4">
                        <div className="flex flex-col">
                           <span className={cn("font-mono font-black text-sm tracking-tight", isOrphaned ? "text-negative" : "text-foreground")}>
                            {pos.symbol}
                          </span>
                        </div>
                      </td>

                      {/* Side */}
                      <td className="px-4 py-4 text-center">
                        <Badge 
                          variant={isLong ? "positive" : "negative"} 
                          className="text-[9px] h-4.5 px-2 font-bold bg-background/40"
                        >
                          {cleanSide}
                        </Badge>
                      </td>

                      {/* Qty */}
                      <td className="px-4 py-4 text-right font-mono font-bold text-foreground">
                        {fmt(pos.qty, 6).replace(/\.?0+$/, "") || "0"}
                      </td>

                      {/* Avg Entry */}
                      <td className="px-4 py-4 text-right font-mono text-muted-foreground">
                        ${fmt(pos.avg_price, 4)}
                      </td>

                      {/* Current */}
                      <td className="px-4 py-4 text-right font-mono text-foreground font-bold">
                        ${fmt(pos.current_price, 4)}
                      </td>

                      {/* Market Value */}
                      <td className="px-4 py-4 text-right font-mono text-foreground font-bold">
                        {fmtValue(pos.market_value)}
                      </td>

                      {/* Unrealized P&L */}
                      <td className={cn("px-4 py-4 text-right font-mono font-black text-sm", plPositive ? "text-positive" : "text-negative")}>
                        {fmtMoney(pos.unrealized_pl)}
                      </td>

                      {/* Return % */}
                      <td className={cn("px-4 py-4 text-right font-mono font-bold text-xs pr-6", getToneClass(plPositive ? "positive" : "negative"))}>
                        {Number.isFinite(pos.unrealized_plpc)
                          ? `${(pos.unrealized_plpc * 100).toFixed(2)}%`
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {orphanCount > 0 && (
        <div className="rounded-2xl border border-negative/30 bg-negative/5 p-4 flex gap-4 animate-in slide-in-from-bottom-2">
           <ShieldAlert className="h-5 w-5 text-negative shrink-0 mt-0.5" />
           <div className="space-y-1">
             <p className="text-xs font-bold text-negative uppercase tracking-wide">Orphaned Position Warning</p>
             <p className="text-[11px] leading-relaxed text-negative/80">
               {orphanCount} broker position{orphanCount > 1 ? "s" : ""} ({fmtValue(totalOrphanMV)} total MV) {orphanCount === 1 ? "is" : "are"} currently unassigned to any APEX strategy. 
               These assets are <span className="font-bold underline">not monitored by risk guardrails</span> and require manual reconciliation or liquidation.
             </p>
           </div>
        </div>
      )}
    </div>
  );
}
