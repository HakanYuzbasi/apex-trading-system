"use client";

import { useState } from "react";
import Link from "next/link";
import {
  BarChart3,
  Bitcoin,
  TrendingUp,
  TrendingDown,
  ArrowRight,
  Wifi,
  WifiOff,
  DollarSign,
  Activity,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";
import { useSessionMetrics, useCockpitData, useBrokerMode, changeBrokerMode } from "@/lib/api";
import { formatCurrency, formatPct, formatCurrencyWithCents } from "@/lib/formatters";
import { SESSION_CONFIG } from "@/lib/constants";
import { useWebSocket } from "@/hooks/useWebSocket";
import AdvancedMetricsPanel from "@/components/dashboard/AdvancedMetricsPanel";
import PitchMetricsRibbon from "@/components/dashboard/PitchMetricsRibbon";
import ShadowTerminal from "@/components/dashboard/ShadowTerminal";
import { ErrorBoundary } from "@/components/ErrorBoundary";

function asNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatMoneyOrDash(value: number | null): string {
  if (value === null) {
    return "—";
  }

  return formatCurrencyWithCents(value);
}

function formatPctOrDash(value: number | null): string {
  if (value === null) {
    return "—";
  }

  return formatPct(value);
}

function formatFixedOrDash(value: number | null, digits = 2): string {
  if (value === null) {
    return "—";
  }

  return value.toFixed(digits);
}

function SessionCard({
  sessionType,
  label,
  description,
  icon: Icon,
  href,
}: {
  sessionType: "core" | "crypto";
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  href: string;
}) {
  const { data: metrics } = useSessionMetrics(sessionType);

  const capital = asNumber(metrics?.capital);
  const sharpe = asNumber(metrics?.sharpe_ratio);
  const dailyPnl = asNumber(metrics?.daily_pnl);
  const winRate = asNumber(metrics?.win_rate);
  const positions = metrics?.open_positions ?? 0;
  const sharpeTarget = SESSION_CONFIG[sessionType].sharpeTarget;
  const sharpeRatio = sharpe ?? 0;
  const sharpeProgress = sharpe !== null
    ? Math.max(8, Math.min(100, (sharpeRatio / Math.max(0.1, Number(sharpeTarget))) * 100))
    : 0;

  return (
    <Link
      href={href}
      className="group flex flex-col gap-4 rounded-2xl border border-border/80 bg-background/70 p-6 transition-all hover:border-primary/40 hover:shadow-lg"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">{label}</h2>
            <p className="text-xs text-muted-foreground">{description}</p>
          </div>
        </div>
        <ArrowRight className="h-5 w-5 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-primary" />
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <p className="text-xs text-muted-foreground">Capital</p>
          <p className="text-sm font-semibold text-foreground">
            {capital !== null ? formatCurrency(capital) : "—"}
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Sharpe Ratio</p>
          <p
            className={`text-sm font-semibold ${
              sharpe !== null && sharpe >= sharpeTarget ? "text-positive" : "text-foreground"
            }`}
          >
            {formatFixedOrDash(sharpe)}{" "}
            <span className="text-xs text-muted-foreground">
              / {sharpeTarget.toFixed(1)}
            </span>
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Daily P&L</p>
          <p
            className={`text-sm font-semibold ${
              (dailyPnl ?? 0) >= 0 ? "text-positive" : "text-negative"
            }`}
          >
            {dailyPnl !== null ? formatCurrency(dailyPnl) : "—"}
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Win Rate</p>
          <p className="text-sm font-semibold text-foreground">
            {formatPctOrDash(winRate)}
          </p>
        </div>
      </div>

      {/* Sharpe progress bar */}
      <div>
        <div className="mb-1 flex justify-between text-xs text-muted-foreground">
          <span>Sharpe Progress</span>
          <span className={sharpe !== null && sharpe < 0 ? "text-red-500 font-bold" : ""}>
            {formatFixedOrDash(sharpe)} / {Number(sharpeTarget).toFixed(1)}
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted/20 relative">
          <div
            className="h-full transition-all duration-1000 ease-out absolute left-0 top-0"
            style={{
              width: `${sharpe !== null ? (sharpe >= 0 ? sharpeProgress : 100) : 0}%`,
              backgroundColor: sharpe !== null && Number(sharpe) >= Number(sharpeTarget) 
                ? "#10b981"  /* Emerald 500 */
                : sharpe !== null && Number(sharpe) < 0 
                  ? "#ef4444" /* Red 500 */
                  : "#3b82f6", /* Blue 500 (Primary) */
              boxShadow: sharpe !== null && Number(sharpe) >= Number(sharpeTarget) ? "0 0 15px rgba(16,185,129,0.5)" : "none",
              borderRadius: "999px"
            }}
          />
        </div>
        {sharpe !== null && Number(sharpe) < 0 && (
          <p className="mt-1 text-[10px] text-red-500/90 font-medium animate-pulse">Critical: Performance below baseline</p>
        )}
      </div>

      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <TrendingUp className="h-3 w-3" />
        <span>{positions} open positions</span>
      </div>
    </Link>
  );
}

function BrokerModeControl() {
  const { data: modeData, mutate } = useBrokerMode();
  const [isChanging, setIsChanging] = useState(false);
  const [modeError, setModeError] = useState<string | null>(null);

  const currentMode = modeData?.broker_mode || "both";

  const handleModeChange = async (target: string) => {
    if (target === currentMode) return;
    setIsChanging(true);
    setModeError(null);
    try {
      await changeBrokerMode(target);
      await mutate();
    } catch (err) {
      // apiJson now extracts FastAPI's {"detail": "..."} into the error message.
      // Strip the leading "API 409: " prefix so the UI just shows the actionable reason.
      const raw = err instanceof Error ? err.message : String(err);
      const msg = raw.replace(/^API \d+:\s*/, "");
      setModeError(msg);
    } finally {
      setIsChanging(false);
    }
  };

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2 rounded-xl border border-border/60 bg-background/40 p-1.5">
        <div className="px-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
          Execution Mode
        </div>
        <div className="flex gap-1">
          {[
            { id: "alpaca", label: "Full Alpaca" },
            { id: "both", label: "Hybrid (Mix)" },
            { id: "ibkr", label: "IBKR Only" },
          ].map((m) => (
            <button
              key={m.id}
              onClick={() => handleModeChange(m.id)}
              disabled={isChanging}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                currentMode === m.id
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              } ${isChanging ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              {isChanging && currentMode === m.id ? (
                <RefreshCw className="h-3 w-3 animate-spin" />
              ) : (
                m.label
              )}
            </button>
          ))}
        </div>
      </div>
      {modeError && (
        <p className="flex items-center gap-1.5 px-2 text-[10px] font-medium text-destructive">
          <AlertTriangle className="h-3 w-3 shrink-0" />
          {modeError}
        </p>
      )}
    </div>
  );
}

/* ─── Portfolio Overview Widget ─── */
function PortfolioOverview({
  isConnected,
}: {
  isConnected: boolean;
}) {
  const { data: cockpitResponse, isLoading } = useCockpitData();
  const { data: sessionMetricsResponse } = useSessionMetrics("core");

  const cockpit = cockpitResponse ?? null;
  const metrics = sessionMetricsResponse ?? null;

  // Final Aggressive Fallbacks for Data Integrity.
  // displayDailyPnl = total MTM daily P&L (unrealized + realized) — same source as Dashboard.tsx
  // so both pages agree on the headline number. daily_pnl_realized is shown as a sub-label only.
  const displayEquity = asNumber(cockpit?.status?.total_equity ?? cockpit?.status?.capital ?? metrics?.capital);
  const displayDailyPnl = asNumber(cockpit?.status?.daily_pnl ?? metrics?.daily_pnl ?? cockpit?.status?.daily_pnl_realized ?? metrics?.daily_pnl_realized);
  const displayRealizedPnl = asNumber((cockpit?.status as unknown as Record<string, unknown>)?.daily_pnl_realized ?? metrics?.daily_pnl_realized);
  const displayTotalPnl = asNumber(cockpit?.status?.total_pnl ?? metrics?.total_pnl);
  const displayPositions = asNumber(cockpit?.status?.open_positions ?? metrics?.open_positions);
  
  const status = cockpit?.status;
  const sharpe = asNumber(status?.sharpe_ratio);
  const winRate = asNumber(status?.win_rate ?? metrics?.win_rate);
  const brokers = (Array.isArray(status?.brokers) ? status.brokers : []);
  const alertCount = cockpit?.alerts?.length ?? 0;
  const criticalAlerts = cockpit?.alerts?.filter((a) => a.severity === "critical").length ?? 0;

  const apiReachable = isConnected || status?.api_reachable;

  if (isLoading) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-24 animate-pulse rounded-xl border border-border/60 bg-muted/30" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Connection status & Controls - PROMINENT RELOCATION */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-2xl border border-border/40 bg-background/50 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="flex flex-col">
            <span className="text-[10px] text-muted-foreground uppercase font-bold tracking-tight">System Status</span>
            <span
              className={`inline-flex items-center gap-2 rounded-full px-2 py-1 text-xs font-semibold ${
                apiReachable
                  ? "bg-positive/15 text-positive"
                  : "bg-destructive/15 text-destructive"
              }`}
            >
              {apiReachable ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
              {apiReachable ? "Linked" : "Offline"}
            </span>
          </div>

          <div className="h-8 w-px bg-border/40 mx-2" />

          <div className="flex flex-col">
            <span className="text-[10px] text-muted-foreground uppercase font-bold tracking-tight">Active Brokers</span>
            <div className="flex gap-2 mt-1">
              {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
              {brokers.map((b: any) => (
                <span
                  key={b.broker}
                  className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                    b.mode === "trading"
                      ? "bg-positive/15 text-positive"
                      : "bg-muted text-muted-foreground/60"
                  }`}
                >
                  <span className={`h-1 w-1 rounded-full ${b.mode === "trading" ? "bg-positive" : "bg-muted-foreground"}`} />
                  {String(b.broker).toUpperCase()}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="flex w-full items-center gap-4 sm:w-auto">
          <BrokerModeControl />
        </div>
      </div>

        {alertCount > 0 && (
          <div className="space-y-2">
            <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ${
              criticalAlerts > 0 ? "bg-negative/15 text-negative" : "bg-warning/15 text-warning"
            }`}>
              <AlertTriangle className="h-3 w-3" />
              {alertCount} alert{alertCount !== 1 ? "s" : ""} active
            </span>
            <div className="flex flex-col gap-1.5 ml-1">
              {(cockpit?.alerts || []).map((alert, i: number) => (
                <div key={alert.id ?? i} className={`text-[10px] flex items-center gap-2 ${
                  alert.severity === "critical" ? "text-negative" : alert.severity === "info" ? "text-muted-foreground" : "text-warning"
                }`}>
                  <span className="h-1 w-1 rounded-full bg-current" />
                  <span className="font-bold uppercase tracking-tight">[{alert.source}]</span>
                  <span>{alert.title}</span>
                  {alert.detail && <span className="text-muted-foreground truncate max-w-[300px]">— {alert.detail}</span>}
                </div>
              ))}
            </div>
          </div>
        )}

      {/* KPI Grid */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="flex items-start gap-3 rounded-xl border border-border/70 bg-background/70 p-4 transition-all hover:border-primary/30 hover:shadow-sm">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10">
            <DollarSign className="h-4 w-4 text-primary" />
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Total Equity</p>
            <p className="text-lg font-bold text-foreground">{formatMoneyOrDash(displayEquity)}</p>
          </div>
        </div>

        <div className="flex items-start gap-3 rounded-xl border border-border/70 bg-background/70 p-4 transition-all hover:border-primary/30 hover:shadow-sm">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10">
            {(displayDailyPnl ?? 0) >= 0 ? <TrendingUp className="h-4 w-4 text-positive" /> : <TrendingDown className="h-4 w-4 text-negative" />}
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Today&apos;s P&L (MTM)</p>
            <p className={`text-lg font-bold ${(displayDailyPnl ?? 0) >= 0 ? "text-positive" : "text-negative"}`}>
              {formatMoneyOrDash(displayDailyPnl)}
            </p>
            {displayRealizedPnl !== null && (
              <p className={`text-xs ${(displayRealizedPnl) >= 0 ? "text-positive/70" : "text-negative/70"}`}>
                Realized: {formatMoneyOrDash(displayRealizedPnl)}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-start gap-3 rounded-xl border border-border/70 bg-background/70 p-4 transition-all hover:border-primary/30 hover:shadow-sm">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10">
            <Activity className="h-4 w-4 text-primary" />
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Open Positions</p>
            <p className="text-lg font-bold text-foreground">{displayPositions ?? "—"}</p>
            <p className="text-xs text-muted-foreground">
              Sharpe: {formatFixedOrDash(sharpe)}
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3 rounded-xl border border-border/70 bg-background/70 p-4 transition-all hover:border-primary/30 hover:shadow-sm">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10">
            <ShieldCheck className="h-4 w-4 text-primary" />
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Win Rate</p>
            <p className="text-lg font-bold text-foreground">{formatPctOrDash(winRate)}</p>
            <p className="text-xs text-muted-foreground">
              {status?.total_trades ?? 0} total trades
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function DashboardOverview() {
  const websocket = useWebSocket();

  return (
    <div className="mx-auto max-w-[1200px] px-4 py-8">
      {/* APEX HARDENED UI MARKER */}
      <script dangerouslySetInnerHTML={{ __html: 'console.log("APEX_HARDENED_UI_V5_ACTIVE")' }} />

      <div className="mb-8">
        <PitchMetricsRibbon
          telemetryMessage={websocket.lastMessage}
          isConnected={websocket.isConnected}
          isConnecting={websocket.isConnecting}
        />
      </div>

      <div className="mb-10">
        <ShadowTerminal
          telemetryMessage={websocket.lastMessage}
          isConnected={websocket.isConnected}
          isConnecting={websocket.isConnecting}
          reconnectAttempt={websocket.reconnectAttempt}
        />
      </div>

      {/* Advanced Analytics Section (Aggressive Top Mounting) */}
      <div id="advanced-analytics-section" className="mb-12 min-h-[500px]">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-primary" />
              Institutional Risk Overlay
            </h2>
            <p className="text-sm text-muted-foreground">
              Live strategy health tracking and alpha decay diagnostics.
            </p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
            </span>
            <span className="text-[10px] font-bold uppercase tracking-widest text-primary">Live Telemetry</span>
          </div>
        </div>
        <div className="rounded-2xl border border-primary/20 bg-background/40 backdrop-blur-sm overflow-hidden shadow-2xl">
          <ErrorBoundary fallback={<div className="p-10 text-center text-muted-foreground">Risk metrics initialization pending...</div>}>
            <AdvancedMetricsPanel />
          </ErrorBoundary>
        </div>
      </div>

      {/* Portfolio Overview */}
      <div className="mb-10">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">
              Portfolio Overview
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Live aggregate metrics across all brokers and strategies.
            </p>
          </div>
        </div>
        <div className="mt-4">
          <PortfolioOverview isConnected={websocket.isConnected} />
        </div>
      </div>

      {/* Trading Sessions */}
      <div className="mb-8">
        <h2 className="text-xl font-bold text-foreground">
          Trading Sessions
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Two independent strategies — Core (equities/forex/indices) and Crypto
          Sleeve (toggleable). Both targeting 1.5+ Sharpe.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <SessionCard
          sessionType="core"
          label="Core Strategy"
          description="Equities, indices, and forex — no crypto"
          icon={BarChart3}
          href="/dashboard/core"
        />
        <SessionCard
          sessionType="crypto"
          label="Crypto Sleeve"
          description="Cryptocurrency trading — toggleable"
          icon={Bitcoin}
          href="/dashboard/crypto"
        />
      </div>

    </div>
  );
}
