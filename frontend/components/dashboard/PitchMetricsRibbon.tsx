"use client";

import { Activity, ShieldAlert, TrendingDown, Wallet, Wifi, WifiOff } from "lucide-react";
import { usePitchMetrics, type PitchMetricsData } from "@/lib/api";
import { type WebSocketMessage } from "@/hooks/useWebSocket";
import { ErrorState } from "@/components/ui/ErrorState";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";
import { Badge } from "@/components/ui/badge";
import { getToneClass } from "@/lib/utils";

type PitchMetricsRibbonProps = {
  telemetryMessage: WebSocketMessage | null;
  isConnected: boolean;
  isConnecting?: boolean;
};

type PitchMetricCardProps = {
  label: string;
  value: string;
  hint: string;
  tone?: "positive" | "negative" | "warning" | "neutral";
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
  if (parsed === null) return null;
  if (Math.abs(parsed) > 100) return null;
  return Math.abs(parsed) > 1 ? parsed / 100 : parsed;
}

function parsePitchMetrics(value: unknown): PitchMetricsData | null {
  if (!isRecord(value)) return null;

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
  if (value === null) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value: number | null): string {
  if (value === null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatSignedMoney(value: number | null): string {
  if (value === null) return "—";
  const formatted = formatMoney(Math.abs(value));
  return value > 0 ? `+${formatted}` : value < 0 ? `-${formatted}` : formatted;
}

function formatFixed(value: number | null, digits = 2): string {
  if (value === null) return "—";
  return value.toFixed(digits);
}

function PitchMetricCard({
  label,
  value,
  hint,
  tone = "neutral",
  icon: Icon,
}: PitchMetricCardProps) {
  return (
    <div className="glass-card flex min-h-[120px] items-start gap-4 rounded-2xl border border-border/10 p-5 transition-all hover:bg-background/50 hover:border-primary/20 hover:scale-[1.02] cursor-default shadow-sm hover:shadow-md duration-300">
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 shadow-[0_0_15px_rgba(var(--primary-rgb),0.1)]">
        <Icon className="h-5 w-5 text-primary" />
      </div>
      <div className="min-w-0 space-y-1">
        <p className="text-[9px] font-black uppercase tracking-widest text-muted-foreground/60">{label.toUpperCase()}</p>
        <p className={`text-2xl font-black font-mono tracking-tight tabular-nums ${getToneClass(tone, "text")}`}>{value}</p>
        <p className="text-[10px] font-black text-muted-foreground uppercase tracking-tighter leading-tight truncate opacity-50">{hint}</p>
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
      <div className="glass-card rounded-2xl p-8 mb-6">
        <LoadingSpinner
          label="Syncing Alpha metrics"
          detail="Deriving realized P&L, margin utilization, and Sharpe from live engine telemetry."
        />
      </div>
    );
  }

  if (!isReady || !metrics) {
    return (
      <div className="glass-card rounded-2xl p-8 mb-6 border-dashed border-muted-foreground/20">
        <ErrorState
          title="Telemetry synchronization pending"
          message={metrics?.error ?? "The dashboard is waiting for derived risk metrics from the active trading hive."}
        />
      </div>
    );
  }

  const marginHint =
    metrics.active_margin !== null && metrics.equity !== null
      ? `${formatMoney(metrics.active_margin)} deployed vs ${formatMoney(metrics.equity)} equity`
      : "Allocation pending";
  const sharpeHint = `${metrics.curve_points} sample points${metrics.sample_interval_seconds ? ` · ${metrics.sample_interval_seconds.toFixed(1)}s cadence` : ""}`;
  const drawdownHint = metrics.source === "ws_stream" ? "Derived from live rolling curve." : `Source: ${metrics.source.replaceAll("_", " ")}`;

  return (
    <section className="space-y-4 animate-in fade-in duration-700">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border/10 pb-4">
        <div className="flex items-center gap-3">
           <div className="px-2 py-0.5 rounded-lg bg-primary/20 text-primary border border-primary/20">
              <span className="text-[10px] font-black uppercase tracking-widest">PITCH HEADERS</span>
           </div>
           <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-tight">
              Institutional risk posture derived from live broker equity and historical curve diagnostics.
           </p>
        </div>
        
        <Badge 
          variant="outline" 
          className={`px-3 py-1 text-[9px] font-black uppercase gap-1.5 transition-all ${
            isConnected ? "bg-positive/10 text-positive border-positive/20" : "bg-warning/10 text-warning border-warning/20 animate-pulse"
          }`}
        >
          {isConnected ? <Wifi size={10} className="text-positive" /> : <WifiOff size={10} className="text-warning" />}
          {isConnected ? "TELEMETRY LIVE" : "POLLING FALLBACK"}
        </Badge>
      </div>

      <div className="grid gap-4 xl:grid-cols-4 md:grid-cols-2">
        <PitchMetricCard
          label="Today's Realized Gain"
          value={formatSignedMoney(metrics.realized_pnl_today)}
          hint="Broker-truth actual intraday realized."
          tone={(metrics.realized_pnl_today ?? 0) >= 0 ? "positive" : "negative"}
          icon={Wallet}
        />
        <PitchMetricCard
          label="Active Margin utilization"
          value={formatPercent(metrics.active_margin_utilization)}
          hint={marginHint}
          tone={(metrics.active_margin_utilization ?? 0) <= 0.5 ? "positive" : "warning"}
          icon={ShieldAlert}
        />
        <PitchMetricCard
          label="Rolling Sharpe (Audit)"
          value={formatFixed(metrics.sharpe_ratio)}
          hint={sharpeHint}
          tone={(metrics.sharpe_ratio ?? 0) >= 1.5 ? "positive" : (metrics.sharpe_ratio ?? 0) < 0 ? "negative" : "warning"}
          icon={Activity}
        />
        <PitchMetricCard
          label="Maximum Drawdown"
          value={formatPercent(metrics.max_drawdown)}
          hint={drawdownHint}
          tone={(metrics.max_drawdown ?? 0) <= -0.1 ? "negative" : "neutral"}
          icon={TrendingDown}
        />
      </div>
    </section>
  );
}
