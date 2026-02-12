"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { MetricCard } from "./MetricCard";
import RiskMetricsRow from "./RiskMetricsRow";
import SectorChart from "./SectorChart";
import AlertNotifications, { AlertItem, AlertSeverity } from "./AlertNotifications";
import ConnectionStatus from "./ConnectionStatus";
import PerformanceCharts from "./PerformanceCharts";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useAuthContext as useAuth } from "@/components/auth/AuthProvider";
import { formatCurrency, formatPct } from "@/lib/utils";
import { Activity, TrendingUp, DollarSign, Zap, AlertTriangle, BarChart3 } from "lucide-react";
import { motion } from "framer-motion";
import dynamic from 'next/dynamic';
import Link from "next/link";

// Dynamic import for 3D component (SSR disabled)
const VolatilitySurface3D = dynamic(() => import('./VolatilitySurface3D'), {
    ssr: false,
    loading: () => <div className="w-full h-full flex items-center justify-center text-muted-foreground text-sm">Loading 3D Surface...</div>
});

interface Position {
    qty: number;
    side: string;
    avg_price: number;
    current_price: number;
    pnl: number;
    pnl_pct: number;
    signal_direction: string;
}

interface EquityPoint {
    timestamp: string;
    equity: number;
    drawdown: number;
    sharpe: number;
}

interface TradingState {
    timestamp: string;
    capital: number;
    initial_capital?: number;
    starting_capital?: number;
    max_positions?: number;
    positions: Record<string, Position>;
    daily_pnl: number;
    total_pnl: number;
    sector_exposure: Record<string, number>;
    open_positions: number;
    total_trades: number;
    sharpe_ratio: number;
    win_rate: number;
    max_drawdown: number;
    // Alert data from backend
    alerts?: Array<{
        id: string;
        severity: AlertSeverity;
        title: string;
        message: string;
        timestamp: string;
        source?: string;
    }>;
    // Equity history for charts
    equity_history?: EquityPoint[];
    // Circuit breaker status
    circuit_breaker_active?: boolean;
    circuit_breaker_reason?: string;
}

type ViewTab = "overview" | "charts";

export default function Dashboard() {
    const [state, setState] = useState<TradingState | null>(null);
    const [alerts, setAlerts] = useState<AlertItem[]>([]);
    const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
    const [activeTab, setActiveTab] = useState<ViewTab>("overview");
    const alertIdCounter = useRef(0);
    const prevCircuitBreaker = useRef(false);

    // Generate unique alert ID
    const nextAlertId = useCallback(() => {
        alertIdCounter.current += 1;
        return `alert-${Date.now()}-${alertIdCounter.current}`;
    }, []);

    // Push a new alert
    const pushAlert = useCallback((severity: AlertSeverity, title: string, message: string, source?: string) => {
        const alert: AlertItem = {
            id: nextAlertId(),
            severity,
            title,
            message,
            timestamp: new Date(),
            source,
        };
        setAlerts(prev => [alert, ...prev].slice(0, 50));
    }, [nextAlertId]);

    // Dismiss alert handlers
    const dismissAlert = useCallback((id: string) => {
        setAlerts(prev => prev.filter(a => a.id !== id));
    }, []);

    const dismissAllAlerts = useCallback(() => {
        setAlerts([]);
    }, []);

    // Handle incoming WebSocket messages
    const handleMessage = useCallback((data: unknown) => {
        const msg = data as Record<string, unknown>;

        if (msg.type === "state_update") {
            const tradingState = msg as unknown as TradingState;
            setState(tradingState);

            // Build equity history from updates
            if (tradingState.capital) {
                setEquityHistory(prev => {
                    const point: EquityPoint = {
                        timestamp: tradingState.timestamp,
                        equity: tradingState.capital, // Backend now provides full liquidation value as capital
                        drawdown: tradingState.max_drawdown ?? 0,
                        sharpe: tradingState.sharpe_ratio ?? 0,
                    };

                    // If backend provides full history, use it
                    if (tradingState.equity_history?.length) {
                        return tradingState.equity_history;
                    }

                    // Otherwise accumulate from updates (keep last 500 points)
                    const updated = [...prev, point];
                    return updated.slice(-500);
                });
            }

            // Check for circuit breaker activation
            if (tradingState.circuit_breaker_active && !prevCircuitBreaker.current) {
                pushAlert(
                    "critical",
                    "Circuit Breaker Activated",
                    tradingState.circuit_breaker_reason || "Trading halted due to risk limits",
                    "RiskManager"
                );
            }
            prevCircuitBreaker.current = tradingState.circuit_breaker_active ?? false;

            // Process backend alerts
            if (tradingState.alerts?.length) {
                for (const a of tradingState.alerts) {
                    pushAlert(a.severity, a.title, a.message, a.source);
                }
            }
        }

        if (msg.type === "alert") {
            const alertMsg = msg as { severity: AlertSeverity; title: string; message: string; source?: string };
            pushAlert(alertMsg.severity, alertMsg.title, alertMsg.message, alertMsg.source);
        }
    }, [pushAlert]);

    const { accessToken: saasToken } = useAuth();
    const apiBaseUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");
    const apiKey = process.env.NEXT_PUBLIC_API_KEY;
    const authToken = saasToken || process.env.NEXT_PUBLIC_AUTH_TOKEN;
    const wsDefault = apiBaseUrl.replace(/^http/, "ws") + "/ws";
    const wsUrlBase = process.env.NEXT_PUBLIC_WS_URL || wsDefault;
    const wsQuery = apiKey ? `api_key=${encodeURIComponent(apiKey)}` :
        authToken ? `token=${encodeURIComponent(authToken)}` : "";
    const wsUrl = wsQuery ? `${wsUrlBase}${wsUrlBase.includes("?") ? "&" : "?"}${wsQuery}` : wsUrlBase;

    const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

    // WebSocket with exponential backoff reconnection
    const { state: wsState, connect: wsConnect, disconnect: wsDisconnect } = useWebSocket({
        url: wsUrl,
        onMessage: handleMessage,
        onConnect: () => {
            pushAlert("success", "Connected", "Live feed established with APEX Trading", "WebSocket");
        },
        onDisconnect: () => {
            pushAlert("warning", "Disconnected", "Lost connection to APEX Trading. Reconnecting...", "WebSocket");
        },
        reconnect: true,
        maxReconnectAttempts: 10,
        initialReconnectDelay: 1000,
        maxReconnectDelay: 30000,
    });

    // Fetch initial state via REST on mount
    useEffect(() => {
        const headers: HeadersInit = {};
        if (apiKey) headers["X-API-Key"] = apiKey;
        if (authToken) headers["Authorization"] = `Bearer ${authToken}`;

        fetch(`${apiBaseUrl}/state`, { headers })
            .then(async res => {
                if (res.status === 401) {
                    pushAlert(
                        "warning",
                        "Auth Required",
                        "Backend rejected /state. Set NEXT_PUBLIC_API_KEY or NEXT_PUBLIC_AUTH_TOKEN.",
                        "API"
                    );
                }
                return res.json();
            })
            .then(data => {
                if (data && data.timestamp) {
                    handleMessage({ type: "state_update", ...data });
                }
            })
            .catch((err) => {
                console.warn("[Dashboard] Failed to fetch initial state:", err.message);
            });
    }, [handleMessage, apiBaseUrl, apiKey, authToken, pushAlert]);

    // Backend health check polling
    useEffect(() => {
        let isMounted = true;
        const check = async () => {
            try {
                const headers: HeadersInit = {};
                if (apiKey) headers["X-API-Key"] = apiKey;
                if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
                const res = await fetch(`${apiBaseUrl}/health`, { headers });
                const data = await res.json();
                if (isMounted) setBackendOnline(data?.status === "online");
            } catch {
                if (isMounted) setBackendOnline(false);
            }
        };
        check();
        const id = setInterval(check, 5000);
        return () => {
            isMounted = false;
            clearInterval(id);
        };
    }, [apiBaseUrl, apiKey, authToken]);

    // Safe defaults
    const capital = state?.capital ?? 0;
    const totalPnl = state?.total_pnl ?? 0;
    const startingCapital = state?.starting_capital ?? state?.initial_capital ?? 0;
    const pnlPct = startingCapital > 0 ? (totalPnl / startingCapital) * 100 : 0;
    const openPositions = state?.open_positions ?? 0;
    const maxPositions = state?.max_positions ?? 40;
    const totalTrades = state?.total_trades ?? 0;
    const positions = state?.positions ?? {};

    // Risk Metrics
    const sharpeRatio = state?.sharpe_ratio ?? 0;
    const winRate = state?.win_rate ?? 0;
    const maxDrawdown = state?.max_drawdown ?? 0;
    const sectorExposure = state?.sector_exposure ?? {};

    return (
        <div className="p-6 h-screen w-full flex flex-col gap-4 overflow-hidden bg-background">
            {/* Alert Notifications (floating) */}
            <AlertNotifications
                alerts={alerts}
                onDismiss={dismissAlert}
                onDismissAll={dismissAllAlerts}
            />

            {/* Header */}
            <header className="flex justify-between items-center glass-panel p-4 rounded-lg shrink-0">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-primary flex items-center justify-center shadow-[0_0_10px_rgba(0,243,255,0.4)]">
                        <Activity className="text-black w-5 h-5" />
                    </div>
                    <h1 className="text-xl font-bold tracking-widest text-glow">APEX <span className="text-primary">TERMINAL</span></h1>
                </div>

                <div className="flex items-center gap-3">
                    {/* View Toggle */}
                    <div className="flex items-center rounded bg-muted/30 border border-border/30 p-0.5">
                        <button
                            onClick={() => setActiveTab("overview")}
                            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${activeTab === "overview"
                                ? "bg-primary/20 text-primary"
                                : "text-muted-foreground hover:text-white"
                                }`}
                        >
                            Overview
                        </button>
                        <button
                            onClick={() => setActiveTab("charts")}
                            className={`px-3 py-1 rounded text-xs font-medium transition-colors flex items-center gap-1.5 ${activeTab === "charts"
                                ? "bg-primary/20 text-primary"
                                : "text-muted-foreground hover:text-white"
                                }`}
                        >
                            <BarChart3 className="w-3 h-3" />
                            Charts
                        </button>
                    </div>

                    {/* Connection Indicator */}
                    <div className="flex items-center gap-2 px-3 py-1 rounded bg-muted/50 border border-border/50">
                        <div className={`w-2 h-2 rounded-full ${wsState.isConnected
                            ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]"
                            : wsState.isConnecting
                                ? "bg-yellow-500 animate-pulse"
                                : "bg-red-500"
                            }`} />
                        <span className="text-xs font-mono text-muted-foreground">
                            {wsState.isConnected ? "LIVE FEED" : wsState.isConnecting ? "CONNECTING..." : "OFFLINE"}
                        </span>
                    </div>
                    {/* Backend Indicator */}
                    <div className="flex items-center gap-2 px-3 py-1 rounded bg-muted/50 border border-border/50">
                        <div className={`w-2 h-2 rounded-full ${backendOnline === null
                            ? "bg-yellow-500 animate-pulse"
                            : backendOnline
                                ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]"
                                : "bg-red-500"
                            }`} />
                        <span className="text-xs font-mono text-muted-foreground">
                            {backendOnline === null ? "BACKEND…" : backendOnline ? "BACKEND OK" : "BACKEND DOWN"}
                        </span>
                    </div>
                    {/* Auth links */}
                    {saasToken ? (
                        <Link href="/settings" className="text-xs text-muted-foreground hover:text-foreground transition">Settings</Link>
                    ) : (
                        <Link href="/login" className="text-xs text-primary hover:underline">Sign in</Link>
                    )}
                </div>
            </header>

            {/* Connection Status Banner (shown only when disconnected) */}
            <ConnectionStatus wsState={wsState} onReconnect={wsConnect} />

            {/* Offline Banner (shown when backend health reports offline/stale) */}
            {backendOnline === false && (
                <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    className="shrink-0 flex items-center justify-between px-4 py-2 rounded-lg text-sm bg-red-950/60 border border-red-500/40 text-red-300"
                >
                    <div className="flex items-center gap-3">
                        <AlertTriangle className="w-4 h-4 text-red-400" />
                        <span className="font-medium">System Offline</span>
                        <span className="text-xs opacity-70">Data shown may be stale. Backend is unreachable or trading state has not been updated.</span>
                    </div>
                </motion.div>
            )}

            {/* Main Wrapper - dim content when backend is offline */}
            <div className={`flex-1 flex flex-col gap-6 min-h-0 overflow-y-auto pr-2 transition-opacity duration-300 ${backendOnline === false ? "opacity-50 pointer-events-none select-none" : ""}`}>

                {/* Top KPI Row (always visible) */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 shrink-0">
                    <MetricCard
                        title="Total Equity"
                        value={formatCurrency(capital)}
                        subValue={`${totalPnl >= 0 ? '+' : ''}${formatCurrency(totalPnl)} P&L`}
                        trend={totalPnl >= 0 ? "up" : "down"}
                        icon={<DollarSign className="w-4 h-4" />}
                        glowColor={totalPnl >= 0 ? "#00f3ff" : "#ff3333"}
                    />
                    <MetricCard
                        title="P&L %"
                        value={`${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%`}
                        subValue="Total Return"
                        trend={pnlPct >= 0 ? "up" : "down"}
                        icon={<TrendingUp className="w-4 h-4" />}
                        glowColor={pnlPct >= 0 ? "#00ff9d" : "#ff3333"}
                    />
                    <MetricCard
                        title="Open Positions"
                        value={openPositions}
                        subValue={`of ${maxPositions} max`}
                        trend={openPositions > maxPositions * 0.8 ? "neutral" : "up"}
                        icon={<Activity className="w-4 h-4" />}
                    />
                    <MetricCard
                        title="Total Trades"
                        value={totalTrades}
                        subValue="Session Trades"
                        icon={<Zap className="w-4 h-4" />}
                    />
                </div>

                {/* Risk Metrics Row */}
                <RiskMetricsRow
                    sharpeRatio={sharpeRatio}
                    winRate={winRate}
                    maxDrawdown={maxDrawdown}
                />

                {/* Tab Content */}
                {activeTab === "overview" ? (
                    /* Overview Tab - Original Layout */
                    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 flex-1 min-h-[400px]">

                        {/* Main Chart Area (Volatility Surface) - 5 Cols */}
                        <motion.div
                            className="col-span-12 md:col-span-5 glass-panel rounded-xl relative overflow-hidden flex flex-col"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 0.1 }}
                        >
                            <div className="absolute top-4 left-6 z-10">
                                <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">3D Volatility Surface</h2>
                            </div>
                            <div className="flex-1 w-full relative">
                                <VolatilitySurface3D />
                            </div>
                        </motion.div>

                        {/* Sector Chart - 3 Cols */}
                        <motion.div
                            className="col-span-12 md:col-span-3 h-full"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.15 }}
                        >
                            <SectorChart data={sectorExposure} />
                        </motion.div>

                        {/* Side Panel (Positions) - 4 Cols */}
                        <motion.div
                            className="col-span-12 md:col-span-4 glass-panel rounded-xl p-6 overflow-hidden flex flex-col"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.2 }}
                        >
                            <h3 className="text-sm font-medium text-muted-foreground mb-4 uppercase flex justify-between items-center">
                                <span>Active Positions</span>
                                <span className="text-xs bg-muted px-2 py-0.5 rounded-full">{Object.keys(positions).length}</span>
                            </h3>

                            <div className="space-y-1 overflow-y-auto pr-2 custom-scrollbar flex-1">
                                {Object.entries(positions).length === 0 ? (
                                    <div className="text-muted-foreground text-center py-10 flex flex-col items-center gap-2">
                                        <AlertTriangle className="w-8 h-8 opacity-20" />
                                        <span>No active positions</span>
                                    </div>
                                ) : (
                                    Object.entries(positions).map(([symbol, pos]) => (
                                        <div key={symbol} className="flex justify-between items-center p-3 rounded hover:bg-white/5 transition-colors border-b border-white/5 last:border-0">
                                            <div className="flex flex-col">
                                                <div className="flex items-center gap-2">
                                                    <span className={`font-bold text-sm ${pos.side === 'LONG' ? 'text-green-400' : 'text-red-400'}`}>
                                                        {pos.side === 'LONG' ? 'L' : 'S'}
                                                    </span>
                                                    <span className="font-bold text-white tracking-wide">{symbol}</span>
                                                </div>
                                                <span className="text-muted-foreground text-[10px] mt-0.5 flex gap-2">
                                                    <span>{pos.qty} sh</span>
                                                    <span>@ ${pos.avg_price?.toFixed(1) ?? '0.0'}</span>
                                                </span>
                                            </div>
                                            <div className="text-right">
                                                <div className="text-white font-mono text-sm">${pos.current_price?.toFixed(2)}</div>
                                                <div className={`text-xs font-mono font-medium ${pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                    {pos.pnl >= 0 ? '+' : ''}{pos.pnl?.toFixed(0)} ({pos.pnl_pct?.toFixed(1)}%)
                                                </div>
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </motion.div>
                    </div>
                ) : (
                    /* Charts Tab - Performance Charts */
                    <PerformanceCharts
                        equityHistory={equityHistory}
                        initialCapital={startingCapital}
                    />
                )}
            </div>
        </div>
    );
}
