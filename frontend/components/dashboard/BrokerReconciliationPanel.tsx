"use client";

import { useMemo } from "react";
import { AlertTriangle, CheckCircle2, RefreshCw } from "lucide-react";

interface BrokerPosition {
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
    <div className="flex flex-col gap-3">
      {/* Header summary */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 rounded-xl border border-negative/30 bg-negative/10 px-3 py-2">
          <AlertTriangle className="h-4 w-4 text-negative" />
          <span className="text-sm font-semibold text-negative">{orphanCount} Orphaned</span>
          {orphanCount > 0 && (
            <span className="text-xs text-negative/80">({fmtValue(totalOrphanMV)} untracked MV)</span>
          )}
        </div>
        <div className="flex items-center gap-2 rounded-xl border border-positive/30 bg-positive/10 px-3 py-2">
          <CheckCircle2 className="h-4 w-4 text-positive" />
          <span className="text-sm font-semibold text-positive">{trackedCount} Tracked by APEX</span>
        </div>
        {isStale && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <RefreshCw className="h-3.5 w-3.5 animate-spin" />
            Awaiting broker sync…
          </div>
        )}
        {lastUpdated && (
          <span className="text-xs text-muted-foreground">
            Refreshed every 30 s · Last: {String(lastUpdated).split("T")[1]?.replace(/\.\d+.*$/, "") ?? lastUpdated}
          </span>
        )}
      </div>

      {isEmpty ? (
        <div className="rounded-xl border border-border/70 bg-background/60 px-4 py-8 text-center text-sm text-muted-foreground">
          {isStale
            ? "Waiting for first broker position sync (up to 30 s after startup)…"
            : "No open positions at broker. Flat book."}
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-border/70">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/60 bg-background/80">
                <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">Status</th>
                <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">Symbol</th>
                <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">Side</th>
                <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground">Qty</th>
                <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground">Avg Entry</th>
                <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground">Current</th>
                <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground">Mkt Value</th>
                <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground">Unrlzd P&L</th>
                <th className="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground">Return</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos, idx) => {
                const isOrphaned = pos.is_orphaned;
                const plPositive = pos.unrealized_pl >= 0;
                return (
                  <tr
                    key={`${pos.symbol}-${idx}`}
                    className={[
                      "border-b border-border/40 transition-colors",
                      isOrphaned
                        ? "bg-negative/8 hover:bg-negative/12"
                        : "bg-background/40 hover:bg-secondary/30",
                    ].join(" ")}
                  >
                    {/* Status badge */}
                    <td className="px-3 py-2.5">
                      {isOrphaned ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-negative/20 px-2 py-0.5 text-[11px] font-bold uppercase tracking-wide text-negative">
                          <AlertTriangle className="h-3 w-3" />
                          ORPHANED
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-positive/15 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-positive">
                          <CheckCircle2 className="h-3 w-3" />
                          TRACKED
                        </span>
                      )}
                    </td>

                    {/* Symbol */}
                    <td className="px-3 py-2.5">
                      <span className={`font-mono font-semibold ${isOrphaned ? "text-negative" : "text-foreground"}`}>
                        {pos.symbol}
                      </span>
                      {isOrphaned && (
                        <p className="text-[10px] text-muted-foreground">not in any active APEX pair</p>
                      )}
                    </td>

                    {/* Side */}
                    <td className="px-3 py-2.5">
                      {(() => {
                        // Normalize raw enum strings like "PositionSide.LONG" → "LONG"
                        const rawSide = String(pos.side ?? "").toUpperCase();
                        const cleanSide = rawSide
                          .replace(/^POSITIONSIDE\./i, "")
                          .replace(/^POSITION_SIDE\./i, "")
                          .trim() || "UNKNOWN";
                        const isLong = cleanSide === "LONG" || cleanSide === "BUY";
                        return (
                          <span className={`rounded px-1.5 py-0.5 text-[11px] font-semibold uppercase ${
                            isLong
                              ? "bg-positive/10 text-positive"
                              : "bg-negative/10 text-negative"
                          }`}>
                            {cleanSide}
                          </span>
                        );
                      })()}
                    </td>

                    {/* Qty */}
                    <td className="px-3 py-2.5 text-right font-mono text-foreground">
                      {fmt(pos.qty, 6).replace(/\.?0+$/, "") || "0"}
                    </td>

                    {/* Avg Entry */}
                    <td className="px-3 py-2.5 text-right font-mono text-muted-foreground">
                      ${fmt(pos.avg_price, 4)}
                    </td>

                    {/* Current */}
                    <td className="px-3 py-2.5 text-right font-mono text-foreground">
                      ${fmt(pos.current_price, 4)}
                    </td>

                    {/* Market Value */}
                    <td className="px-3 py-2.5 text-right font-mono text-foreground">
                      {fmtValue(pos.market_value)}
                    </td>

                    {/* Unrealized P&L */}
                    <td className={`px-3 py-2.5 text-right font-mono font-semibold ${plPositive ? "text-positive" : "text-negative"}`}>
                      {fmtMoney(pos.unrealized_pl)}
                    </td>

                    {/* Return % */}
                    <td className={`px-3 py-2.5 text-right font-mono text-xs ${plPositive ? "text-positive" : "text-negative"}`}>
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
      )}

      {orphanCount > 0 && (
        <p className="rounded-lg border border-negative/25 bg-negative/8 px-3 py-2 text-xs text-negative/90">
          <strong>Action required:</strong> {orphanCount} broker position{orphanCount > 1 ? "s" : ""} ({fmtValue(totalOrphanMV)} total market value) exist{orphanCount === 1 ? "s" : ""} at Alpaca but {orphanCount > 1 ? "are" : "is"} not tracked by any active APEX strategy pair.
          These positions will not be automatically closed by the risk manager. Close them manually via the Alpaca dashboard or add them to an active strategy pair.
        </p>
      )}
    </div>
  );
}
