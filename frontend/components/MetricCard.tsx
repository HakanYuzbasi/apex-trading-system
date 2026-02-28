"use client";

import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface MetricCardProps {
    title: string;
    value: string | number;
    subValue?: string;
    trend?: "up" | "down" | "neutral";
    icon?: ReactNode;
    className?: string;
    glowColor?: string;
}

export function MetricCard({ title, value, subValue, trend, icon, className, glowColor }: MetricCardProps) {
    return (
        <div
            className={cn(
                "glass-card p-6 rounded-xl flex flex-col justify-between relative overflow-hidden transition-all duration-200 group",
                className
            )}
        >
            {glowColor && (
                <div
                    className="absolute -top-10 -right-10 w-24 h-24 rounded-full blur-3xl opacity-20 pointer-events-none group-hover:opacity-30 transition-opacity duration-200"
                    style={{ backgroundColor: glowColor }}
                />
            )}

            <div className="flex justify-between items-start mb-2 relative z-10">
                <span className="text-muted-foreground text-sm font-medium uppercase tracking-wider">{title}</span>
                {icon && <div className="text-muted-foreground transition-transform duration-200 group-hover:scale-110">{icon}</div>}
            </div>

            <div className="flex items-end gap-2 relative z-10">
                <span className="text-2xl font-bold tracking-tight text-foreground">{value}</span>
            </div>

            {subValue && (
                <div className={cn(
                    "text-xs mt-1 font-medium relative z-10",
                    trend === 'up' ? 'text-positive' : trend === 'down' ? 'text-negative' : 'text-muted-foreground'
                )}>
                    {subValue}
                </div>
            )}
        </div>
    );
}
