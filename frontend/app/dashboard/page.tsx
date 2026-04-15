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
  LayoutDashboard,
} from "lucide-react";
import { useSessionMetrics, useCockpitData, useBrokerMode, changeBrokerMode } from "@/lib/api";
import { formatCurrency, formatPct, formatCurrencyWithCents } from "@/lib/formatters";
import { SESSION_CONFIG } from "@/lib/constants";
import { useWebSocket } from "@/hooks/useWebSocket";
import AdvancedMetricsPanel from "@/components/dashboard/AdvancedMetricsPanel";
import PitchMetricsRibbon from "@/components/dashboard/PitchMetricsRibbon";
import ShadowTerminal from "@/components/dashboard/ShadowTerminal";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Badge } from "@/components/ui/badge";
import { getToneClass } from "@/lib/utils";

function asNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatMoneyOrDash(value: number | null): string {
  if (value === null) return "—";
  return formatCurrencyWithCents(value);
}

function formatPctOrDash(value: number | null): string {
  if (value === null) return "—";
  return formatPct(value);
}

function formatFixedOrDash(value: number | null, digits = 2): string {
  if (value === null) return "—";
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
  
  const tone = (dailyPnl ?? 0) >= 0 ? "positive" : "negative";

  return (
    <Link
      href={href}
      className="group glass-card flex flex-col gap-6 rounded-3xl border border-border/10 p-7 transition-all hover:bg-background/40 hover:scale-[1.01] hover:shadow-2xl duration-500"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10 border border-primary/20 shadow-[0_0_20px_rgba(59,130,246,0.1)]">
            <Icon className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h2 className="text-xl font-black text-foreground uppercase tracking-tight leading-none">{label}</h2>
            <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest mt-1 opacity-60">{description}</p>
          </div>
        </div>
        <div className="h-10 w-10 flex items-center justify-center rounded-full border border-border/10 bg-background/20 group-hover:bg-primary group-hover:text-primary-foreground transition-all duration-500">
           <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-0.5" />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6 sm:grid-cols-4">
        <div className="space-y-1">
          <p className="text-[9px] font-black text-muted-foreground uppercase tracking-widest leading-none">Capital</p>
          <p className="text-base font-black font-mono leading-none tracking-tight text-foreground">
            {capital !== null ? formatCurrency(capital) : "—"}
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-[9px] font-black text-muted-foreground uppercase tracking-widest leading-none">Sharpe Ratio</p>
          <p className={`text-base font-black font-mono leading-none tracking-tight ${sharpe !== null && sharpe >= sharpeTarget ? "text-positive" : "text-foreground"}`}>
            {formatFixedOrDash(sharpe)}
            <span className="text-[10px] font-black text-muted-foreground/40 ml-1">/ {sharpeTarget.toFixed(1)}</span>
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-[9px] font-black text-muted-foreground uppercase tracking-widest leading-none">Daily P&L</p>
          <p className={`text-base font-black font-mono leading-none tracking-tight ${getToneClass(tone, "text")}`}>
            {dailyPnl !== null ? formatCurrency(dailyPnl) : "—"}
          </p>
        </div>
        <div className="space-y-1">
          <p className="text-[9px] font-black text-muted-foreground uppercase tracking-widest leading-none">Win Rate</p>
          <p className="text-base font-black font-mono leading-none tracking-tight text-foreground">
            {formatPctOrDash(winRate)}
          </p>
        </div>
      </div>

      <div className="h-px bg-border/10 w-full" />

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 group/pos">
          <div className="h-2 w-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
          <span className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.15em] leading-none">{positions} ACTIVE POSITIONS</span>
        </div>
        <Badge variant="outline" className="text-[9px] font-black bg-background/20 py-0.5 px-2 tracking-tighter">TARGET: {sharpeTarget.toFixed(1)} SHARPE</Badge>
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
      const raw = err instanceof Error ? err.message : String(err);
      const msg = raw.replace(/^API \d+:\s*/, "");
      setModeError(msg);
    } finally {
      setIsChanging(false);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 rounded-2xl border border-border/10 bg-background/20 p-1.5 shadow-inner">
        <div className="px-3 text-[9px] font-black uppercase tracking-[0.2em] text-muted-foreground/60 border-r border-border/10">
          EXECUTION
        </div>
        <div className="flex gap-1">
          {[
            { id: "alpaca", label: "ALPACA" },
            { id: "both", label: "HYBRID" },
            { id: "ibkr", label: "IBKR" },
          ].map((m) => (
            <button
              key={m.id}
              onClick={() => handleModeChange(m.id)}
              disabled={isChanging}
              className={`rounded-xl px-4 py-2 text-[10px] font-black tracking-widest transition-all duration-300 ${
                currentMode === m.id
                  ? "bg-primary text-primary-foreground shadow-[0_0_20px_rgba(59,130,246,0.3)]"
                  : "text-muted-foreground hover:bg-background/40 hover:text-foreground"
              } ${isChanging ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              {isChanging && currentMode === m.id ? (
                <RefreshCw className="h-3 w-3 animate-spin mx-auto" />
              ) : (
                m.label
              )}
            </button>
          ))}
        </div>
      </div>
      {modeError && (
        <p className="flex items-center gap-1.5 px-3 text-[9px] font-black uppercase tracking-tighter text-negative animate-bounce">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
          CRITICAL: {modeError}
        </p>
      )}
    </div>
  );
}

function PortfolioOverview({
  isConnected,
}: {
  isConnected: boolean;
}) {
  const { data: cockpitResponse, isLoading } = useCockpitData();
  const { data: sessionMetricsResponse } = useSessionMetrics("core");

  const cockpit = cockpitResponse ?? null;
  const metrics = sessionMetricsResponse ?? null;

  const displayEquity = asNumber(cockpit?.status?.total_equity ?? cockpit?.status?.capital ?? metrics?.capital);
  const displayDailyPnl = asNumber(cockpit?.status?.daily_pnl ?? metrics?.daily_pnl ?? cockpit?.status?.daily_pnl_realized ?? metrics?.daily_pnl_realized);
  const displayRealizedPnl = asNumber((cockpit?.status as unknown as Record<string, unknown>)?.daily_pnl_realized ?? metrics?.daily_pnl_realized);
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
          <div key={i} className="h-32 animate-pulse rounded-[2rem] border border-border/10 bg-background/20" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-6 p-6 rounded-[2rem] border border-border/10 bg-background/20 backdrop-blur-3xl shadow-2xl relative overflow-hidden group">
        <div className="absolute inset-x-0 h-px top-0 bg-gradient-to-r from-transparent via-primary/20 to-transparent opacity-50" />
        
        <div className="flex items-center gap-6">
          <div className="flex flex-col border-l-2 border-primary/40 pl-4">
            <span className="text-[10px] text-muted-foreground uppercase font-black tracking-[0.2em] leading-none mb-2">SYSTEM TELEMETRY</span>
            <div className="flex items-center gap-3">
               <Badge 
                 className={`gap-2 py-1 px-3 text-[10px] font-black border-2 transition-all duration-700 ${
                   apiReachable ? "bg-positive/10 text-positive border-positive/20" : "bg-negative/10 text-negative border-negative/20 animate-pulse"
                 }`}
               >
                 {apiReachable ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
                 {apiReachable ? "LINKED" : "OFFLINE"}
               </Badge>
               
               <div className="flex gap-2">
                 {brokers.map((b: any) => (
                   <span
                     key={b.broker}
                     className={`flex items-center gap-1.5 px-3 py-1 rounded-full border border-border/10 text-[9px] font-black uppercase tracking-tighter ${
                       b.mode === "trading" ? "bg-positive/5 text-positive/80" : "bg-background/40 text-muted-foreground/40"
                     }`}
                   >
                     <span className={`h-1 w-1 rounded-full ${b.mode === "trading" ? "bg-positive animate-pulse" : "bg-muted-foreground/30"}`} />
                     {String(b.broker).toUpperCase()}
                   </span>
                 ))}
               </div>
            </div>
          </div>
        </div>

        <div className="flex w-full items-center gap-6 sm:w-auto">
          <BrokerModeControl />
        </div>
      </div>

      {alertCount > 0 && (
        <div className="p-5 rounded-2xl border border-negative/20 bg-negative/5 animate-in slide-in-from-top-4 duration-500">
          <div className="flex items-center gap-2 mb-4">
             <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-negative/20 text-negative shadow-[0_0_15px_rgba(239,68,68,0.2)]">
                <AlertTriangle className="h-4 w-4" />
             </div>
             <div>
                <h4 className="text-[10px] font-black text-negative uppercase tracking-widest leading-none">{alertCount} ACTIVE PRODUCTION ALERTS</h4>
                <p className="text-[9px] text-negative/60 font-medium uppercase tracking-tighter mt-1">Institutional monitoring engine reporting discrepancies</p>
             </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {(cockpit?.alerts || []).map((alert, i: number) => (
              <div key={alert.id ?? i} className={`p-2.5 rounded-xl border flex items-start gap-4 transition-all hover:bg-background/20 ${
                alert.severity === "critical" ? "border-negative/20 bg-negative/5 text-negative" : "border-warning/20 bg-warning/5 text-warning"
              }`}>
                <div className="flex flex-col gap-0.5 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded bg-background/40">[{alert.source}]</span>
                    <span className="text-[10px] font-black uppercase truncate">{alert.title}</span>
                  </div>
                  {alert.detail && <p className="text-[10px] text-muted-foreground font-medium truncate opacity-60">— {alert.detail}</p>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "TOTAL ASSET EQUITY", value: formatMoneyOrDash(displayEquity), icon: DollarSign, primary: true },
          { 
            label: "INTRADAY MTM P&L", 
            value: formatMoneyOrDash(displayDailyPnl), 
            sub: displayRealizedPnl !== null ? `REALIZED: ${formatMoneyOrDash(displayRealizedPnl)}` : null, 
            tone: (displayDailyPnl ?? 0) >= 0 ? "positive" : "negative",
            icon: (displayDailyPnl ?? 0) >= 0 ? TrendingUp : TrendingDown 
          },
          { 
            label: "ACTIVE POSITIONS", 
            value: String(displayPositions ?? "—"), 
            sub: `SHARPE: ${formatFixedOrDash(sharpe)}`, 
            icon: Activity 
          },
          { 
            label: "EXECUTION WIN RATE", 
            value: formatPctOrDash(winRate), 
            sub: `${status?.total_trades ?? 0} TRADES COMMITTED`, 
            icon: ShieldCheck 
          },
        ].map((kpi, idx) => (
          <div key={idx} className="glass-card flex flex-col gap-4 rounded-[2rem] border border-border/10 p-6 transition-all hover:scale-[1.03] hover:shadow-2xl duration-500">
             <div className="flex items-center justify-between">
                <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${kpi.tone ? getToneClass(kpi.tone as any, "bg") : "bg-primary/10"} border border-border/10`}>
                   <kpi.icon className={`h-5 w-5 ${kpi.tone ? getToneClass(kpi.tone as any, "text") : "text-primary"}`} />
                </div>
                <div className="w-10 h-px bg-border/20" />
             </div>
             <div className="space-y-1">
                <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest leading-none">{kpi.label}</p>
                <p className={`text-2xl font-black font-mono tracking-tight leading-none pt-2 ${kpi.tone ? getToneClass(kpi.tone as any, "text") : "text-foreground"}`}>
                   {kpi.value}
                </p>
                {kpi.sub && <p className="text-[10px] font-black text-muted-foreground uppercase tracking-tight pt-1 opacity-50">{kpi.sub}</p>}
             </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function DashboardOverview() {
  const websocket = useWebSocket();

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-12 space-y-16 animate-in fade-in duration-1000">
      <script dangerouslySetInnerHTML={{ __html: 'console.log("APEX_PREMIUM_UI_HARDENED_V6")' }} />

      {/* Header Section */}
      <header className="flex flex-wrap items-end justify-between gap-8 border-b border-border/10 pb-10">
        <div className="space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20">
             <LayoutDashboard size={14} className="text-primary" />
             <span className="text-[10px] font-black uppercase tracking-[0.2em] text-primary">APEX TERMINAL SYSTEM</span>
          </div>
          <div className="space-y-1">
             <h1 className="text-4xl font-black text-foreground uppercase tracking-tighter sm:text-5xl">GLOBAL OVERSIGHT</h1>
             <p className="text-sm text-muted-foreground font-black uppercase tracking-widest max-w-2xl leading-relaxed opacity-60">
                Aggregated institutional-grade telemetry across independent strategy nodes and decentralized broker clusters.
             </p>
          </div>
        </div>
        
        <div className="flex flex-col items-end gap-3 text-right">
           <div className="flex gap-2">
              <Badge variant="outline" className="text-[10px] font-black uppercase bg-background/40 py-1.5 px-4 tracking-widest border-border/20 shadow-xl">ALPHA_v3.6.4</Badge>
              <Badge variant="outline" className="text-[10px] font-black uppercase bg-background/40 py-1.5 px-4 tracking-widest border-border/20 shadow-xl">PROD_ENV</Badge>
           </div>
           <p className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em] opacity-40">UTC: {new Date().toISOString().slice(11, 16)} SYNC</p>
        </div>
      </header>

      {/* Core Alpha Metrics */}
      <div className="space-y-6">
        <PitchMetricsRibbon
          telemetryMessage={websocket.lastMessage}
          isConnected={websocket.isConnected}
          isConnecting={websocket.isConnecting}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-10">
        {/* Risk Overlay - Primary Focus */}
        <div className="xl:col-span-8 space-y-8">
           <div className="flex items-center justify-between">
              <div className="space-y-1 border-l-4 border-primary pl-5">
                 <h2 className="text-2xl font-black text-foreground uppercase tracking-tighter">INSTITUTIONAL RISK OVERSIGHT</h2>
                 <p className="text-[11px] font-black text-muted-foreground uppercase tracking-widest opacity-60">High-fidelity tail risk diagnostics and alpha decay matrix</p>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 rounded-2xl bg-primary/5 border border-primary/10 group hover:border-primary/30 transition-all">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                </span>
                <span className="text-[10px] font-black uppercase tracking-widest text-primary">Neural Stream Pulse</span>
              </div>
           </div>
           
           <ErrorBoundary fallback={<div className="glass-card p-16 rounded-[2rem] text-center text-muted-foreground uppercase font-black tracking-widest">Risk engine cold start in progress...</div>}>
              <AdvancedMetricsPanel />
           </ErrorBoundary>
        </div>

        {/* Audit Log / Terminal */}
        <div className="xl:col-span-4 space-y-8">
           <div className="space-y-1 border-l-4 border-muted-foreground/40 pl-5">
              <h2 className="text-2xl font-black text-foreground uppercase tracking-tighter text-muted-foreground/80">SHADOW AUDIT</h2>
              <p className="text-[11px] font-black text-muted-foreground uppercase tracking-widest opacity-60">PPO suggested execution layer</p>
           </div>
           <ShadowTerminal
             telemetryMessage={websocket.lastMessage}
             isConnected={websocket.isConnected}
             isConnecting={websocket.isConnecting}
             reconnectAttempt={websocket.reconnectAttempt}
           />
        </div>
      </div>

      {/* Portfolio Aggregation Section */}
      <section className="space-y-10 pt-10 border-t border-border/10">
        <div className="space-y-1 border-l-4 border-foreground pl-5">
          <h2 className="text-3xl font-black text-foreground uppercase tracking-tighter">PORTFOLIO AGGREGATION</h2>
          <p className="text-[11px] font-black text-muted-foreground uppercase tracking-widest opacity-60">Unified broker reconciliation and equity curve projection across all active mandates.</p>
        </div>
        
        <PortfolioOverview isConnected={websocket.isConnected} />
      </section>

      {/* Mandate Entry Points */}
      <section className="space-y-10 pt-10 border-t border-border/10">
        <div className="space-y-1 border-l-4 border-primary pl-5">
          <h2 className="text-3xl font-black text-foreground uppercase tracking-tighter">INDEPENDENT STRATEGY MANDATES</h2>
          <p className="text-[11px] font-black text-muted-foreground uppercase tracking-widest opacity-60">High-frequency execution nodes separated by asset class and regime bias.</p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <SessionCard
            sessionType="core"
            label="Core Strategy Mandate"
            description="Institutional Equity, Forex, and Global Indices"
            icon={BarChart3}
            href="/dashboard/core"
          />
          <SessionCard
            sessionType="crypto"
            label="Crypto Sleeve Mandate"
            description="Systematic Digital Asset Arbitrage and Directional Bias"
            icon={Bitcoin}
            href="/dashboard/crypto"
          />
        </div>
      </section>
      
      <footer className="pt-20 pb-10 flex items-center justify-center opacity-30">
         <div className="flex items-center gap-3">
            <span className="h-px w-12 bg-border" />
            <span className="text-[10px] font-black tracking-[0.5em] uppercase text-muted-foreground">APEX MULTI-VENUE EXECUTION NODE</span>
            <span className="h-px w-12 bg-border" />
         </div>
      </footer>
    </div>
  );
}
