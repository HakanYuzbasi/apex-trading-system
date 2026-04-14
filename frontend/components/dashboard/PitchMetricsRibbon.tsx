"use client";

import { Activity, ShieldAlert, TrendingDown, Wallet } from "lucide-react";
import { usePitchMetrics, type PitchMetricsData } from "@/lib/api";
import { type WebSocketMessage } from "@/hooks/useWebSocket";
import { ErrorState } from "@/components/ui/ErrorState";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

type PitchMetricsRibbonProps = {
  telemetryMessage: WebSocketMessage | null;
  isConnected: boolean;
  isConnecting?: boolean;
};

type PitchMetricCardProps = {
  label: string;
  value: string;
  hint: string;
  tone?: "positive" | "negative" | "neutral";
  icon: React.ComponentType<{ className?: string }>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function asNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeDrawdown(value: unknown): number | null {
  const parsed = asNumber(value);
  if (parsed === null) {
    return null;
  }

  if (Math.abs(parsed) > 100) {
    return null;
  }

  return Math.abs(parsed) > 1 ? parsed / 100 : parsed;
}

function parsePitchMetrics(value: unknown): PitchMetricsData | null {
  if (!isRecord(value)) {
    return null;
  }

  return {
    available: Boolean(value.available),
    error: typeof value.error === "string" ? value.error : null,
    timestamp: typeof value.timestamp === "string" ? value.timestamp : null,
    source: typeof value.source === "string" ? value.source : "unknown",
    equity: asNumber(value.equity),
    realized_pnl_today: asNumber(value.realized_pnl_today),
    active_margin: asNumber(value.active_margin),
    active_margin_utilization: asNumber(value.active_margin_utilization),
    sharpe_ratio: asNumber(value.sharpe_ratio),
    max_drawdown: normalizeDrawdown(value.max_drawdown),
    curve_points: Number.isFinite(Number(value.curve_points)) ? Number(value.curve_points) : 0,
    sample_interval_seconds: asNumber(value.sample_interval_seconds),
  };
}

function formatMoney(value: number | null): string {
  if (value === null) {
    return "—";
  }

  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value: number | null): string {
  if (value === null) {
    return "—";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function formatSignedMoney(value: number | null): string {
  if (value === null) {
    return "—";
  }

  const formatted = formatMoney(Math.abs(value));
  return value > 0 ? `+${formatted}` : value < 0 ? `-${formatted.replace("-", "")}` : formatted;
}

function formatFixed(value: number | null, digits = 2): string {
  if (value === null) {
    return "—";
  }

  return value.toFixed(digits);
}

function PitchMetricCard({
  label,
  value,
  hint,
  tone = "neutral",
  icon: Icon,
}: PitchMetricCardProps) {
  const toneClass =
    tone === "positive"
      ? "text-emerald-300"
      : tone === "negative"
        ? "text-rose-300"
        : "text-zinc-100";

  return (
    <div className="flex min-h-[124px] items-start gap-3 rounded-2xl border border-zinc-800/80 bg-zinc-950/90 px-4 py-4 shadow-[0_18px_40px_rgba(0,0,0,0.35)] transition-colors hover:border-cyan-500/30">
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-cyan-500/20 bg-cyan-500/10">
        <Icon className="h-4 w-4 text-cyan-300" />
      </div>
      <div className="min-w-0 space-y-1">
        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-500">{label}</p>
        <p className={`text-2xl font-semibold tabular-nums ${toneClass}`}>{value}</p>
        <p className="text-xs leading-5 text-zinc-400">{hint}</p>
      </div>
    </div>
  );
}

export default function PitchMetricsRibbon({
  telemetryMessage,
  isConnected,
  isConnecting = false,
}: PitchMetricsRibbonProps) {
  const { data, isLoading } = usePitchMetrics();
  const websocketMetrics =
    telemetryMessage?.type === "state_update" && (telemetryMessage.tenant_id === "unified" || !telemetryMessage.tenant_id)
      ? parsePitchMetrics(telemetryMessage.pitch_metrics)
      : null;
  const metrics = websocketMetrics?.available ? websocketMetrics : data ?? websocketMetrics;
  const isReady = Boolean(metrics?.available);

  if ((isLoading || isConnecting) && !metrics) {
    return (
      <LoadingSpinner
        label="Building investor metrics"
        detail="Deriving realized P&L, margin utilization, Sharpe, and drawdown from live engine telemetry."
        className="min-h-[180px]"
      />
    );
  }

  if (!isReady || !metrics) {
    return (
      <ErrorState
        title="Pitch metrics unavailable"
        message={metrics?.error ?? "The dashboard could not derive investor-ready risk metrics from the current backend state."}
        className="min-h-[180px]"
      />
    );
  }

  const marginHint =
    metrics.active_margin !== null && metrics.equity !== null
      ? `${formatMoney(metrics.active_margin)} deployed against ${formatMoney(metrics.equity)} equity`
      : "Margin allocation unavailable";
  const sharpeHint = `${metrics.curve_points} equity points${metrics.sample_interval_seconds ? ` · ${metrics.sample_interval_seconds.toFixed(1)}s median cadence` : ""}`;
  const drawdownHint = metrics.source === "ws_stream" ? "Derived from the live rolling equity curve." : `Source: ${metrics.source.replaceAll("_", " ")}`;

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-sm font-semibold uppercase tracking-[0.22em] text-zinc-400">Pitch Metrics</h1>
          <p className="mt-1 text-sm text-zinc-500">
            Real-time risk posture derived from live broker equity, engine state, and the historical equity curve.
          </p>
        </div>
        <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] ${
          isConnected
            ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
            : "border-amber-500/30 bg-amber-500/10 text-amber-300"
        }`}>
          <span className={`h-2 w-2 rounded-full ${isConnected ? "bg-emerald-300" : "bg-amber-300"}`} />
          {isConnected ? "WebSocket Live" : "Polling Fallback"}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-4 md:grid-cols-2">
        <PitchMetricCard
          label="Today's Realized PnL"
          value={formatSignedMoney(metrics.realized_pnl_today)}
          hint="Broker-truth realized intraday performance."
          tone={(metrics.realized_pnl_today ?? 0) >= 0 ? "positive" : "negative"}
          icon={Wallet}
        />
        <PitchMetricCard
          label="Active Margin Utilization"
          value={formatPercent(metrics.active_margin_utilization)}
          hint={marginHint}
          tone={(metrics.active_margin_utilization ?? 0) <= 0.5 ? "positive" : "neutral"}
          icon={ShieldAlert}
        />
        <PitchMetricCard
          label="Sharpe Ratio"
          value={formatFixed(metrics.sharpe_ratio)}
          hint={sharpeHint}
          tone={(metrics.sharpe_ratio ?? 0) >= 1 ? "positive" : (metrics.sharpe_ratio ?? 0) < 0 ? "negative" : "neutral"}
          icon={Activity}
        />
        <PitchMetricCard
          label="Max Drawdown"
          value={formatPercent(metrics.max_drawdown)}
          hint={drawdownHint}
          tone={(metrics.max_drawdown ?? 0) <= -0.1 ? "negative" : "neutral"}
          icon={TrendingDown}
        />
      </div>
    </section>
  );
}
