"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import {
    AreaChart, Area, LineChart, Line, XAxis, YAxis, Tooltip,
    ResponsiveContainer, ReferenceLine, CartesianGrid,
} from "recharts";

interface EquityPoint {
    timestamp: string;
    equity: number;
    drawdown: number;
    sharpe: number;
}

interface PerformanceChartsProps {
    equityHistory: EquityPoint[];
    initialCapital: number;
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; dataKey: string }>; label?: string }) {
    if (!active || !payload?.length) return null;

    return (
        <div className="glass-panel rounded-lg p-2 border border-border/50 text-xs text-white">
            <p className="text-muted-foreground mb-1">{label}</p>
            {payload.map((entry, i) => {
                const isDrawdown = entry.dataKey === "drawdown";
                const isSharpe = entry.dataKey === "sharpe";
                const value = isDrawdown
                    ? `${(entry.value * 100).toFixed(2)}%`
                    : isSharpe
                        ? entry.value.toFixed(2)
                        : `$${entry.value.toLocaleString()}`;
                return (
                    <p key={i} className="font-mono">
                        {entry.dataKey}: {value}
                    </p>
                );
            })}
        </div>
    );
}

export default function PerformanceCharts({ equityHistory, initialCapital }: PerformanceChartsProps) {
    const chartData = useMemo(() => {
        if (!equityHistory?.length) return [];

        return equityHistory.map(point => ({
            time: new Date(point.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            equity: Math.round(point.equity * 100) / 100,
            drawdown: point.drawdown,
            sharpe: Math.round(point.sharpe * 100) / 100,
            pnlPct: ((point.equity - initialCapital) / initialCapital) * 100,
        }));
    }, [equityHistory, initialCapital]);

    const latestPnl = chartData.length > 0 ? chartData[chartData.length - 1].pnlPct : 0;
    const equityColor = latestPnl >= 0 ? "#00f3ff" : "#ff3333";
    const equityGradientId = "equityGradient";
    const drawdownGradientId = "drawdownGradient";

    if (chartData.length < 2) {
        return (
            <motion.div
                className="glass-panel rounded-xl p-6 flex items-center justify-center h-[300px]"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
            >
                <p className="text-muted-foreground text-sm">Waiting for equity history data...</p>
            </motion.div>
        );
    }

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Equity Curve / P&L Chart */}
            <motion.div
                className="glass-panel rounded-xl p-4 flex flex-col"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 }}
            >
                <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                        Equity Curve
                    </h3>
                    <span className={`text-xs font-mono font-medium ${latestPnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                        {latestPnl >= 0 ? "+" : ""}{latestPnl.toFixed(2)}%
                    </span>
                </div>
                <div className="h-[220px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                            <defs>
                                <linearGradient id={equityGradientId} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={equityColor} stopOpacity={0.3} />
                                    <stop offset="95%" stopColor={equityColor} stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                                dataKey="time"
                                tick={{ fontSize: 10, fill: "#666" }}
                                axisLine={false}
                                tickLine={false}
                                interval="preserveStartEnd"
                            />
                            <YAxis
                                tick={{ fontSize: 10, fill: "#666" }}
                                axisLine={false}
                                tickLine={false}
                                tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                                width={50}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <ReferenceLine
                                y={initialCapital}
                                stroke="rgba(255,255,255,0.15)"
                                strokeDasharray="4 4"
                            />
                            <Area
                                type="monotone"
                                dataKey="equity"
                                stroke={equityColor}
                                strokeWidth={2}
                                fill={`url(#${equityGradientId})`}
                                dot={false}
                                animationDuration={800}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </motion.div>

            {/* Drawdown Chart */}
            <motion.div
                className="glass-panel rounded-xl p-4 flex flex-col"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
            >
                <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                        Drawdown
                    </h3>
                    {chartData.length > 0 && (
                        <span className="text-xs font-mono text-red-400">
                            Max: {(Math.min(...chartData.map(d => d.drawdown)) * 100).toFixed(2)}%
                        </span>
                    )}
                </div>
                <div className="h-[220px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                            <defs>
                                <linearGradient id={drawdownGradientId} x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#ff3333" stopOpacity={0.4} />
                                    <stop offset="95%" stopColor="#ff3333" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                                dataKey="time"
                                tick={{ fontSize: 10, fill: "#666" }}
                                axisLine={false}
                                tickLine={false}
                                interval="preserveStartEnd"
                            />
                            <YAxis
                                tick={{ fontSize: 10, fill: "#666" }}
                                axisLine={false}
                                tickLine={false}
                                tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
                                width={50}
                                domain={["dataMin", 0]}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <ReferenceLine
                                y={-0.08}
                                stroke="#ff6b6b"
                                strokeDasharray="4 4"
                                label={{ value: "Circuit Breaker", fill: "#ff6b6b", fontSize: 10, position: "insideTopRight" }}
                            />
                            <Area
                                type="monotone"
                                dataKey="drawdown"
                                stroke="#ff3333"
                                strokeWidth={1.5}
                                fill={`url(#${drawdownGradientId})`}
                                dot={false}
                                animationDuration={800}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </motion.div>

            {/* Rolling Sharpe Ratio Chart */}
            <motion.div
                className="glass-panel rounded-xl p-4 flex flex-col col-span-1 md:col-span-2"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.35 }}
            >
                <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                        Rolling Sharpe Ratio
                    </h3>
                    {chartData.length > 0 && (
                        <span className={`text-xs font-mono font-medium ${chartData[chartData.length - 1].sharpe > 1 ? "text-green-400" : chartData[chartData.length - 1].sharpe < 0 ? "text-red-400" : "text-yellow-400"}`}>
                            Current: {chartData[chartData.length - 1].sharpe.toFixed(2)}
                        </span>
                    )}
                </div>
                <div className="h-[180px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                                dataKey="time"
                                tick={{ fontSize: 10, fill: "#666" }}
                                axisLine={false}
                                tickLine={false}
                                interval="preserveStartEnd"
                            />
                            <YAxis
                                tick={{ fontSize: 10, fill: "#666" }}
                                axisLine={false}
                                tickLine={false}
                                width={35}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
                            <ReferenceLine
                                y={2}
                                stroke="#00ff9d"
                                strokeDasharray="4 4"
                                label={{ value: "Target", fill: "#00ff9d", fontSize: 10, position: "insideTopRight" }}
                            />
                            <Line
                                type="monotone"
                                dataKey="sharpe"
                                stroke="#00f3ff"
                                strokeWidth={2}
                                dot={false}
                                animationDuration={800}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </motion.div>
        </div>
    );
}
