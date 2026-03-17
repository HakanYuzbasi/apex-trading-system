"use client";

import { useMemo, useState } from "react";
import {
  BarChart3,
  Bitcoin,
  DollarSign,
  Gauge,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  useSessionMetrics,
  useSessionPositions,
  type CockpitPosition,
} from "@/lib/api";
import { SESSION_CONFIG, type SessionType } from "@/lib/constants";
import {
  formatCurrency,
  formatCurrencyWithCents,
  formatPct,
  normalizeDrawdownPct,
  sortIndicator,
} from "@/lib/formatters";
import { Skeleton } from "@/components/ui/skeleton";

type SortKey = "symbol" | "qty" | "entry" | "current" | "pnl" | "pnl_pct" | "signal_direction";
type SortDir = "asc" | "desc";

function MetricTile({
  label,
  value,
  subValue,
  icon: Icon,
  tone,
}: {
  label: string;
  value: string;
  subValue?: string;
  icon: React.ElementType;
  tone?: "positive" | "negative" | "neutral";
}) {
  const color =
    tone === "positive"
      ? "text-positive"
      : tone === "negative"
        ? "text-negative"
        : "text-foreground";

  return (
    <div className="flex items-start gap-3 rounded-xl border border-border/70 bg-background/70 p-4">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10">
        <Icon className="h-4 w-4 text-primary" />
      </div>
      <div className="min-w-0">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className={`text-lg font-bold ${color}`}>{value}</p>
        {subValue && (
          <p className="text-xs text-muted-foreground">{subValue}</p>
        )}
      </div>
    </div>
  );
}

export default function SessionDashboard({
  sessionType,
}: {
  sessionType: SessionType;
}) {
  const cfg = SESSION_CONFIG[sessionType];
  const { data: metrics, isLoading: metricsLoading } =
    useSessionMetrics(sessionType);
  const { data: posData, isLoading: posLoading } =
    useSessionPositions(sessionType);

  const [sortKey, setSortKey] = useState<SortKey>("pnl");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const positions = useMemo(
    () => posData?.positions ?? [],
    [posData?.positions]
  );

  const sortedPositions = useMemo(() => {
    const sorted = [...positions];
    sorted.sort((a, b) => {
      if (sortKey === "symbol" || sortKey === "signal_direction") {
        return String(a[sortKey]).localeCompare(String(b[sortKey]));
      }
      return Number(a[sortKey]) - Number(b[sortKey]);
    });
    if (sortDir === "desc") sorted.reverse();
    return sorted;
  }, [positions, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  // Derived metrics
  const capital = metrics?.capital ?? cfg.initialCapital;
  const startingCapital = metrics?.starting_capital ?? cfg.initialCapital;
  const dailyPnl = metrics?.daily_pnl ?? 0;
  const totalPnl = metrics?.total_pnl ?? 0;
  const sharpe = metrics?.sharpe_ratio ?? 0;
  const winRate = metrics?.win_rate ?? 0;
  const drawdown = metrics?.max_drawdown ?? 0;
  const totalTrades = metrics?.total_trades ?? 0;

  const SessionIcon = sessionType === "crypto" ? Bitcoin : BarChart3;

  if (metricsLoading && posLoading) {
    return (
      <div className="mx-auto max-w-[1600px] px-4 py-6">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1600px] px-4 py-6">
      {/* Header */}
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
          <SessionIcon className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-foreground">{cfg.label}</h1>
          <p className="text-xs text-muted-foreground">{cfg.description}</p>
        </div>
        {sessionType === "crypto" && (
          <span className="ml-auto rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
            Sleeve — Toggleable
          </span>
        )}
      </div>

      {/* Sharpe Target Banner */}
      <div className="mb-6 rounded-xl border border-border/70 bg-background/70 p-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm font-medium text-foreground">
            Sharpe Ratio Target: {cfg.sharpeTarget.toFixed(1)}
          </span>
          <span
            className={`text-sm font-bold ${
              sharpe >= cfg.sharpeTarget ? "text-positive" : "text-foreground"
            }`}
          >
            Current: {sharpe.toFixed(2)}
          </span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              sharpe >= cfg.sharpeTarget ? "bg-positive" : "bg-primary"
            }`}
            style={{
              width: `${Math.min(100, Math.max(0, (sharpe / cfg.sharpeTarget) * 100))}%`,
            }}
          />
        </div>
      </div>

      {/* Metric Tiles */}
      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricTile
          label="Capital"
          value={formatCurrency(capital)}
          subValue={`Started: ${formatCurrency(startingCapital)}`}
          icon={DollarSign}
        />
        <MetricTile
          label="Daily P&L"
          value={formatCurrencyWithCents(dailyPnl)}
          tone={dailyPnl >= 0 ? "positive" : "negative"}
          icon={dailyPnl >= 0 ? TrendingUp : TrendingDown}
        />
        <MetricTile
          label="Total P&L"
          value={formatCurrencyWithCents(totalPnl)}
          subValue={`${startingCapital > 0 ? formatPct(totalPnl / startingCapital) : "0%"} return`}
          tone={totalPnl >= 0 ? "positive" : "negative"}
          icon={Gauge}
        />
        <MetricTile
          label="Win Rate"
          value={formatPct(winRate)}
          subValue={`${totalTrades} total trades`}
          icon={ShieldCheck}
        />
      </div>

      {/* Secondary Metrics */}
      <div className="mb-6 grid gap-3 sm:grid-cols-3">
        <div className="rounded-xl border border-border/70 bg-background/70 p-4">
          <p className="text-xs text-muted-foreground">Sharpe Ratio</p>
          <p className="text-2xl font-bold text-foreground">
            {sharpe.toFixed(2)}
          </p>
          <p className="text-xs text-muted-foreground">
            Target: {cfg.sharpeTarget.toFixed(1)}
          </p>
        </div>
        <div className="rounded-xl border border-border/70 bg-background/70 p-4">
          <p className="text-xs text-muted-foreground">Max Drawdown</p>
          <p className="text-2xl font-bold text-negative">
            {normalizeDrawdownPct(drawdown).toFixed(1)}%
          </p>
          <p className="text-xs text-muted-foreground">Budget: 10%</p>
        </div>
        <div className="rounded-xl border border-border/70 bg-background/70 p-4">
          <p className="text-xs text-muted-foreground">Open Positions</p>
          <p className="text-2xl font-bold text-foreground">
            {positions.length}
          </p>
          <p className="text-xs text-muted-foreground">
            Max: {cfg.maxPositions}
          </p>
        </div>
      </div>

      {/* Positions Table */}
      <div className="rounded-2xl border border-border/80 bg-background/70 p-5">
        <h2 className="mb-4 text-sm font-semibold text-foreground">
          Positions ({positions.length})
        </h2>

        {positions.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No open positions in {cfg.label.toLowerCase()}.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-border/50 text-xs text-muted-foreground">
                  {(
                    [
                      ["symbol", "Symbol"],
                      ["qty", "Qty"],
                      ["entry", "Entry"],
                      ["current", "Current"],
                      ["pnl", "P&L"],
                      ["pnl_pct", "P&L %"],
                      ["signal_direction", "Signal"],
                    ] as [SortKey, string][]
                  ).map(([key, label]) => (
                    <th
                      key={key}
                      className="cursor-pointer px-2 py-2 font-medium hover:text-foreground"
                      onClick={() => toggleSort(key)}
                    >
                      {label}
                      {sortIndicator(sortKey === key, sortDir)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedPositions.map((pos) => (
                  <tr
                    key={`${pos.symbol}-${pos.qty}`}
                    className="border-b border-border/30 transition-colors hover:bg-muted/30"
                  >
                    <td className="px-2 py-2 font-medium text-foreground">
                      {pos.symbol}
                    </td>
                    <td className="px-2 py-2 font-mono text-foreground">
                      {pos.qty}
                    </td>
                    <td className="px-2 py-2 font-mono text-foreground">
                      {formatCurrencyWithCents(pos.entry)}
                    </td>
                    <td className="px-2 py-2 font-mono text-foreground">
                      {formatCurrencyWithCents(pos.current)}
                    </td>
                    <td
                      className={`px-2 py-2 font-mono font-semibold ${
                        pos.pnl >= 0 ? "text-positive" : "text-negative"
                      }`}
                    >
                      {formatCurrencyWithCents(pos.pnl)}
                    </td>
                    <td
                      className={`px-2 py-2 font-mono ${
                        pos.pnl_pct >= 0 ? "text-positive" : "text-negative"
                      }`}
                    >
                      {pos.pnl_pct.toFixed(2)}%
                    </td>
                    <td className="px-2 py-2">
                      <span
                        className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                          pos.signal_direction === "LONG" || pos.signal_direction === "BUY"
                            ? "bg-positive/15 text-positive"
                            : pos.signal_direction === "SHORT" || pos.signal_direction === "SELL"
                              ? "bg-negative/15 text-negative"
                              : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {pos.signal_direction}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
