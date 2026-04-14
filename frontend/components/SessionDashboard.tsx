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
} from "@/lib/api";
import { SESSION_CONFIG, type SessionType } from "@/lib/constants";
import {
  formatCurrency,
  formatCurrencyWithCents,
  formatPct,
  normalizeDrawdownPct,
  sortIndicator,
} from "@/lib/formatters";
import { ErrorState } from "@/components/ui/ErrorState";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

type SortKey = "symbol" | "qty" | "entry" | "current" | "pnl" | "pnl_pct" | "signal_direction";
type SortDir = "asc" | "desc";

function formatMoneyOrDash(value: number | null | undefined): string {
  return value == null ? "—" : formatCurrencyWithCents(value);
}

function formatPctOrDash(value: number | null | undefined): string {
  return value == null ? "—" : formatPct(value);
}

function formatFixedOrDash(value: number | null | undefined, digits = 2): string {
  return value == null ? "—" : Number(value).toFixed(digits);
}

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
  icon: React.ComponentType<{ className?: string }>;
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
  const { data: metrics, isLoading: metricsLoading, error: metricsFetchError } =
    useSessionMetrics(sessionType);
  const { data: posData, isLoading: posLoading, error: positionsFetchError } =
    useSessionPositions(sessionType);

  const [sortKey, setSortKey] = useState<SortKey>("pnl");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const positions = useMemo(
    () => posData?.positions ?? [],
    [posData?.positions]
  );

  const metricsError =
    metrics?.error ??
    (metricsFetchError instanceof Error ? metricsFetchError.message : null);
  const positionsError =
    posData?.error ??
    (positionsFetchError instanceof Error ? positionsFetchError.message : null);
  const hasRenderableData = Boolean(metrics?.available || posData?.available || positions.length > 0);

  const sortedPositions = useMemo(() => {
    const sorted = [...positions];
    sorted.sort((a, b) => {
      if (sortKey === "symbol" || sortKey === "signal_direction") {
        return String(a?.[sortKey] ?? "").localeCompare(String(b?.[sortKey] ?? ""));
      }
      return Number(a?.[sortKey] ?? 0) - Number(b?.[sortKey] ?? 0);
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
  const capital = metrics?.capital ?? null;
  const startingCapital = metrics?.starting_capital ?? metrics?.initial_capital ?? cfg.initialCapital;
  const dailyPnl = metrics?.daily_pnl ?? null;
  const totalPnl = metrics?.total_pnl ?? null;
  const sharpe = metrics?.sharpe_ratio ?? null;
  const winRate = metrics?.win_rate ?? null;
  const drawdown = metrics?.max_drawdown ?? null;
  const totalTrades = metrics?.total_trades ?? 0;

  const SessionIcon = sessionType === "crypto" ? Bitcoin : BarChart3;

  if ((metricsLoading || posLoading) && !hasRenderableData) {
    return (
      <div className="mx-auto max-w-[1600px] px-4 py-6">
        <LoadingSpinner
          label={`Loading ${cfg.label}`}
          detail="Pulling session-scoped metrics and positions through the dashboard BFF."
        />
      </div>
    );
  }

  if (!hasRenderableData && (metricsError || positionsError)) {
    return (
      <div className="mx-auto max-w-[1600px] px-4 py-6">
        <ErrorState
          title={`${cfg.label} unavailable`}
          message={metricsError ?? positionsError ?? "Session telemetry is temporarily unavailable."}
          onRetry={() => window.location.reload()}
        />
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

      {(metricsError || positionsError) && (
        <div className="mb-6 rounded-2xl border border-amber-500/25 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          {metricsError ?? positionsError}
        </div>
      )}

      {/* Sharpe Target Banner */}
      <div className="mb-6 rounded-xl border border-border/70 bg-background/70 p-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm font-medium text-foreground">
            Sharpe Ratio Target: {cfg.sharpeTarget.toFixed(1)}
          </span>
          <span
            className={`text-sm font-bold ${
              sharpe !== null && sharpe >= cfg.sharpeTarget ? "text-positive" : "text-foreground"
            }`}
          >
            Current: {formatFixedOrDash(sharpe)}
          </span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              sharpe !== null && sharpe >= cfg.sharpeTarget ? "bg-positive" : "bg-primary"
            }`}
            style={{
              width: sharpe === null
                ? "0%"
                : `${Math.min(100, Math.max(0, (sharpe / cfg.sharpeTarget) * 100))}%`,
            }}
          />
        </div>
      </div>

      {/* Metric Tiles */}
      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricTile
          label="Capital"
          value={capital !== null ? formatCurrency(capital) : "—"}
          subValue={`Started: ${formatCurrency(startingCapital)}`}
          icon={DollarSign}
        />
        <MetricTile
          label="Daily P&L"
          value={formatMoneyOrDash(dailyPnl)}
          tone={(dailyPnl ?? 0) >= 0 ? "positive" : "negative"}
          icon={(dailyPnl ?? 0) >= 0 ? TrendingUp : TrendingDown}
        />
        <MetricTile
          label="Total P&L"
          value={formatMoneyOrDash(totalPnl)}
          subValue={totalPnl !== null && startingCapital > 0 ? `${formatPct(totalPnl / startingCapital)} return` : "Return unavailable"}
          tone={(totalPnl ?? 0) >= 0 ? "positive" : "negative"}
          icon={Gauge}
        />
        <MetricTile
          label="Win Rate"
          value={formatPctOrDash(winRate)}
          subValue={`${totalTrades} total trades`}
          icon={ShieldCheck}
        />
      </div>

      {/* Secondary Metrics */}
      <div className="mb-6 grid gap-3 sm:grid-cols-3">
        <div className="rounded-xl border border-border/70 bg-background/70 p-4">
          <p className="text-xs text-muted-foreground">Sharpe Ratio</p>
          <p className="text-2xl font-bold text-foreground">
            {formatFixedOrDash(sharpe)}
          </p>
          <p className="text-xs text-muted-foreground">
            Target: {cfg.sharpeTarget.toFixed(1)}
          </p>
        </div>
        <div className="rounded-xl border border-border/70 bg-background/70 p-4">
          <p className="text-xs text-muted-foreground">Max Drawdown</p>
          <p className="text-2xl font-bold text-negative">
            {drawdown !== null ? `${normalizeDrawdownPct(drawdown).toFixed(1)}%` : "—"}
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
          posData?.available === false && positionsError ? (
            <ErrorState
              title="Positions unavailable"
              message={positionsError}
              onRetry={() => window.location.reload()}
              className="min-h-[180px]"
            />
          ) : (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No open positions in {cfg.label.toLowerCase()}.
            </p>
          )
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
                      {pos?.symbol ?? "—"}
                    </td>
                    <td className="px-2 py-2 font-mono text-foreground">
                      {pos?.qty ?? "—"}
                    </td>
                    <td className="px-2 py-2 font-mono text-foreground">
                      {formatMoneyOrDash(pos?.entry ?? null)}
                    </td>
                    <td className="px-2 py-2 font-mono text-foreground">
                      {formatMoneyOrDash(pos?.current ?? null)}
                    </td>
                    <td
                      className={`px-2 py-2 font-mono font-semibold ${
                        (pos?.pnl ?? 0) >= 0 ? "text-positive" : "text-negative"
                      }`}
                    >
                      {formatMoneyOrDash(pos?.pnl ?? null)}
                    </td>
                    <td
                      className={`px-2 py-2 font-mono ${
                        (pos?.pnl_pct ?? 0) >= 0 ? "text-positive" : "text-negative"
                      }`}
                    >
                      {pos?.pnl_pct != null ? `${pos.pnl_pct.toFixed(2)}%` : "—"}
                    </td>
                    <td className="px-2 py-2">
                      <span
                        className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                          pos?.signal_direction === "LONG" || pos?.signal_direction === "BUY"
                            ? "bg-positive/15 text-positive"
                            : pos?.signal_direction === "SHORT" || pos?.signal_direction === "SELL"
                              ? "bg-negative/15 text-negative"
                              : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {pos?.signal_direction ?? "UNKNOWN"}
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
