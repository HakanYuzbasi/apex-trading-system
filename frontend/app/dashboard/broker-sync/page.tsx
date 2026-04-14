"use client";

import { useState, useEffect, useMemo } from "react";
import { useCockpitData } from "@/lib/api";
import BrokerReconciliationPanel from "@/components/dashboard/BrokerReconciliationPanel";
import { ErrorState } from "@/components/ui/ErrorState";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";
import { ShieldAlert, DollarSign, TrendingUp, AlertTriangle, CheckCircle2 } from "lucide-react";

type BrokerPositionRow = {
  symbol: string;
  normalized_symbol: string;
  qty: number;
  side: string;
  market_value: number | null;
  unrealized_pl: number | null;
  unrealized_plpc: number | null;
  current_price: number | null;
  avg_price: number | null;
  is_orphaned: boolean;
};

type BrokerPositionSnapshot = {
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
};

type BrokerSyncResponse = {
  available: boolean;
  broker_positions: BrokerPositionRow[];
  active_pairs: string[];
  timestamp?: string | null;
  total?: number;
  tracked?: number;
  orphaned?: number;
  error?: string | null;
};

function normalizeBrokerPosition(row: BrokerPositionRow): BrokerPositionSnapshot {
  return {
    symbol: row.symbol,
    normalized_symbol: row.normalized_symbol,
    qty: row.qty,
    side: row.side,
    market_value: row.market_value ?? 0,
    unrealized_pl: row.unrealized_pl ?? 0,
    unrealized_plpc: row.unrealized_plpc ?? 0,
    current_price: row.current_price ?? 0,
    avg_price: row.avg_price ?? 0,
    is_orphaned: row.is_orphaned,
  };
}

function fmtMoney(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return "$" + Math.abs(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export default function BrokerSyncPage() {
  const { data: cockpit } = useCockpitData();
  const [brokerData, setBrokerData] = useState<BrokerSyncResponse | null>(null);
  const [brokerError, setBrokerError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Poll for broker positions since WS on port 8000 doesn't receive engine data (usually on 8765)
  useEffect(() => {
    let mounted = true;
    const fetchBrokerPositions = async () => {
      if (!mounted) return;
      try {
        const res = await fetch("/api/v1/broker-positions");
        const json = await res.json().catch(() => null);
        if (!mounted) return;

        if (!res.ok || !json || typeof json !== "object") {
          setBrokerError(`Broker sync upstream returned ${res.status}.`);
          return;
        }

        const payload = json as BrokerSyncResponse;
        setBrokerData(payload);
        setBrokerError(typeof payload.error === "string" ? payload.error : null);
      } catch (err) {
        if (mounted) {
          setBrokerError(err instanceof Error ? err.message : "Failed to fetch broker positions.");
        }
      } finally {
        if (mounted) {
          setIsLoading(false);
        }
      }
    };

    fetchBrokerPositions();
    const interval = setInterval(fetchBrokerPositions, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const brokerPositions = useMemo(
    () => (brokerData?.broker_positions ?? []).map(normalizeBrokerPosition),
    [brokerData?.broker_positions]
  );
  const lastUpdated = brokerData?.timestamp ?? null;

  // Compute summary stats from broker positions
  const stats = useMemo(() => {
    if (!Array.isArray(brokerPositions) || brokerPositions.length === 0) {
      return { total: 0, orphaned: 0, tracked: 0, totalMV: 0, totalPnl: 0, orphanMV: 0 };
    }
    let orphaned = 0, tracked = 0, totalMV = 0, totalPnl = 0, orphanMV = 0;
    for (const p of brokerPositions) {
      const mv = Number(p.market_value) || 0;
      const pnl = Number(p.unrealized_pl) || 0;
      totalMV += mv;
      totalPnl += pnl;
      if (p.is_orphaned) {
        orphaned++;
        orphanMV += mv;
      } else {
        tracked++;
      }
    }
    return { total: brokerPositions.length, orphaned, tracked, totalMV, totalPnl, orphanMV };
  }, [brokerPositions]);

  const positions = cockpit?.positions?.length ?? 0;

  if (isLoading && !brokerData) {
    return (
      <div className="mx-auto max-w-[1600px] px-4 py-6">
        <LoadingSpinner
          label="Loading broker reconciliation"
          detail="Reconciling live broker inventory against the active APEX strategy graph."
        />
      </div>
    );
  }

  if (!brokerData && brokerError) {
    return (
      <div className="mx-auto max-w-[1600px] px-4 py-6">
        <ErrorState
          title="Broker reconciliation unavailable"
          message={brokerError}
          onRetry={() => window.location.reload()}
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-6">
      {/* Header */}
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/15">
          <ShieldAlert className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-foreground">Broker Reconciliation</h1>
          <p className="text-sm text-muted-foreground">
            Raw Alpaca broker positions vs active APEX strategy pairs. Orphaned positions are not managed by the risk engine.
          </p>
        </div>
        <span className="ml-auto rounded-full border border-border/60 bg-muted/50 px-2.5 py-1 text-xs font-medium text-muted-foreground">
          Auto-refresh · 30s
        </span>
      </div>

      {brokerError && (
        <div className="mb-6 rounded-2xl border border-amber-500/25 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          {brokerError}
        </div>
      )}

      {/* Summary Stats */}
      {stats.total > 0 && (
        <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-border/70 bg-background/70 p-4">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="h-4 w-4 text-primary" />
              <p className="text-xs text-muted-foreground">Total Market Value</p>
            </div>
            <p className="text-lg font-bold text-foreground">{fmtMoney(stats.totalMV)}</p>
            <p className="text-xs text-muted-foreground">{stats.total} positions at broker</p>
          </div>

          <div className="rounded-xl border border-border/70 bg-background/70 p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="h-4 w-4 text-primary" />
              <p className="text-xs text-muted-foreground">Unrealized P&L</p>
            </div>
            <p className={`text-lg font-bold ${stats.totalPnl >= 0 ? "text-positive" : "text-negative"}`}>
              {stats.totalPnl >= 0 ? "+" : "-"}{fmtMoney(Math.abs(stats.totalPnl))}
            </p>
          </div>

          <div className="rounded-xl border border-border/70 bg-background/70 p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="h-4 w-4 text-negative" />
              <p className="text-xs text-muted-foreground">Orphaned Positions</p>
            </div>
            <p className="text-lg font-bold text-negative">{stats.orphaned}</p>
            <p className="text-xs text-muted-foreground">{fmtMoney(stats.orphanMV)} untracked</p>
          </div>

          <div className="rounded-xl border border-border/70 bg-background/70 p-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle2 className="h-4 w-4 text-positive" />
              <p className="text-xs text-muted-foreground">APEX Tracked</p>
            </div>
            <p className="text-lg font-bold text-positive">{stats.tracked}</p>
            <p className="text-xs text-muted-foreground">
              {positions} in strategy engine
            </p>
          </div>
        </div>
      )}

      {/* All-orphaned context banner */}
      {stats.orphaned > 0 && stats.tracked === 0 && (
        <div className="mb-4 rounded-xl border border-warning/30 bg-warning/10 p-4">
          <p className="text-sm font-semibold text-foreground mb-1">All broker positions are orphaned</p>
          <p className="text-xs text-muted-foreground">
            The APEX risk engine does not currently have any active strategy pairs mapped to these broker positions.
            This is expected during the initial shadow-mode audit period or when the trading engine is paused.
            These positions will not be automatically managed (rebalanced, hedged, or closed) unless added to an active strategy.
          </p>
        </div>
      )}

      {/* Reconciliation Table */}
      <div className="rounded-2xl border border-border/70 bg-background/60 p-5">
        <BrokerReconciliationPanel
          brokerPositions={brokerPositions}
          lastUpdated={lastUpdated}
        />
      </div>
    </div>
  );
}
