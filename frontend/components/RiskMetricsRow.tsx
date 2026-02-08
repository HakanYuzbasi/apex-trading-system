"use client";

import { MetricCard } from "./MetricCard";
import { ShieldAlert, Crosshair, TrendingDown } from "lucide-react";

interface RiskMetricsProps {
    sharpeRatio: number;
    winRate: number;
    maxDrawdown: number;
}

export default function RiskMetricsRow({ sharpeRatio, winRate, maxDrawdown }: RiskMetricsProps) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <MetricCard
                title="Sharpe Ratio"
                value={sharpeRatio.toFixed(2)}
                subValue={sharpeRatio > 2 ? "Excellent" : sharpeRatio > 1 ? "Good" : "Risk High"}
                trend={sharpeRatio > 1.5 ? "up" : sharpeRatio < 1 ? "down" : "neutral"}
                icon={<ShieldAlert className="w-4 h-4" />}
                glowColor={sharpeRatio > 2 ? "#00f3ff" : undefined}
            />
            <MetricCard
                title="Win Rate"
                value={`${(winRate * 100).toFixed(1)}%`}
                subValue="Target: >55%"
                trend={winRate > 0.55 ? "up" : winRate < 0.45 ? "down" : "neutral"}
                icon={<Crosshair className="w-4 h-4" />}
                glowColor={winRate > 0.6 ? "#00ff9d" : undefined}
            />
            <MetricCard
                title="Max Drawdown"
                value={`${(maxDrawdown * 100).toFixed(2)}%`}
                subValue="Peak to Trough"
                trend={maxDrawdown < -0.1 ? "down" : "neutral"}
                icon={<TrendingDown className="w-4 h-4" />}
                glowColor={maxDrawdown < -0.15 ? "#ff3333" : undefined}
            />
        </div>
    );
}
