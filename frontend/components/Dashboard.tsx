"use client";

import { useEffect, useState, useRef } from "react";
import { MetricCard } from "./MetricCard";
import { formatCurrency, formatPct } from "@/lib/utils";
import { Activity, TrendingUp, DollarSign, Zap, Globe, AlertTriangle } from "lucide-react";
import { motion } from "framer-motion";
import dynamic from 'next/dynamic';

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

interface TradingState {
    timestamp: string;
    capital: number;
    positions: Record<string, Position>;
    daily_pnl: number;
    total_pnl: number;
    sector_exposure: Record<string, number>;
    open_positions: number;
    total_trades: number;
}

export default function Dashboard() {
    const [state, setState] = useState<TradingState | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const wsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        // Fetch initial state via REST immediately
        fetch("http://localhost:8000/state")
            .then(res => res.json())
            .then(data => {
                if (data && data.timestamp) {
                    setState({ type: "state_update", ...data });
                }
            })
            .catch(() => {});

        // Connect to WebSocket for live updates
        const connect = () => {
            const ws = new WebSocket("ws://localhost:8000/ws");
            wsRef.current = ws;

            ws.onopen = () => {
                setIsConnected(true);
                console.log("Connected to APEX Stream");
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === "state_update") {
                        setState(msg);
                    }
                } catch (e) {
                    console.error("Parse error", e);
                }
            };

            ws.onclose = () => {
                setIsConnected(false);
                setTimeout(connect, 3000); // Reconnect
            };
        };

        connect();

        return () => {
            wsRef.current?.close();
        };
    }, []);

    // Safe defaults
    const capital = state?.capital ?? 0;
    const totalPnl = state?.total_pnl ?? 0;
    const pnlPct = capital > 0 ? (totalPnl / capital) * 100 : 0;
    const openPositions = state?.open_positions ?? 0;
    const totalTrades = state?.total_trades ?? 0;
    const positions = state?.positions ?? {};

    return (
        <div className="p-6 h-screen w-full flex flex-col gap-6 overflow-hidden">
            {/* Header */}
            <header className="flex justify-between items-center glass-panel p-4 rounded-lg shrink-0">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-primary flex items-center justify-center">
                        <Activity className="text-black w-5 h-5" />
                    </div>
                    <h1 className="text-xl font-bold tracking-widest text-glow">APEX <span className="text-primary">TERMINAL</span></h1>
                </div>

                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 px-3 py-1 rounded bg-muted/50 border border-border/50">
                        <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]" : "bg-red-500"}`} />
                        <span className="text-xs font-mono text-muted-foreground">{isConnected ? "LIVE FEED" : "OFFLINE"}</span>
                    </div>
                </div>
            </header>

            {/* Main Grid */}
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
                    subValue="of 15 max"
                    trend="neutral"
                    icon={<Activity className="w-4 h-4" />}
                />
                <MetricCard
                    title="Total Trades"
                    value={totalTrades}
                    subValue="Session Trades"
                    icon={<Zap className="w-4 h-4" />}
                />
            </div>

            {/* Content Area */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 flex-1" style={{ minHeight: 400 }}>
                {/* Main Chart Area */}
                <motion.div
                    className="col-span-2 glass-panel rounded-xl relative overflow-hidden"
                    style={{ minHeight: 400 }}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.1 }}
                >
                    <div className="absolute top-4 left-6 z-10">
                        <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">3D Volatility Surface</h2>
                    </div>
                    <div style={{ width: '100%', height: '100%', minHeight: 400 }}>
                        <VolatilitySurface3D />
                    </div>
                </motion.div>

                {/* Side Panel (Positions) */}
                <motion.div
                    className="glass-panel rounded-xl p-6 overflow-auto"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    <h3 className="text-sm font-medium text-muted-foreground mb-4 uppercase">Active Positions</h3>
                    <div className="space-y-3 font-mono text-xs">
                        {Object.entries(positions).length === 0 ? (
                            <div className="text-muted-foreground text-center py-4">No positions</div>
                        ) : (
                            Object.entries(positions).map(([symbol, pos]) => (
                                <div key={symbol} className="flex justify-between items-center py-2 border-b border-white/5 last:border-0">
                                    <div className="flex flex-col">
                                        <span className={`font-bold ${pos.side === 'LONG' ? 'text-green-400' : 'text-red-400'}`}>
                                            {pos.side} {symbol}
                                        </span>
                                        <span className="text-muted-foreground text-[10px]">{pos.qty} shares</span>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-white">${pos.current_price?.toFixed(2)}</div>
                                        <div className={`text-[10px] ${pos.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                            {pos.pnl_pct >= 0 ? '+' : ''}{pos.pnl_pct?.toFixed(1)}%
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
