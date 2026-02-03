"use client";

import { motion } from "framer-motion";
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
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn(
                "glass-panel p-6 rounded-xl flex flex-col justify-between relative overflow-hidden",
                className
            )}
        >
            {glowColor && (
                <div
                    className="absolute -top-10 -right-10 w-24 h-24 rounded-full blur-3xl opacity-20 pointer-events-none"
                    style={{ backgroundColor: glowColor }}
                />
            )}

            <div className="flex justify-between items-start mb-2">
                <span className="text-muted-foreground text-sm font-medium uppercase tracking-wider">{title}</span>
                {icon && <div className="text-muted-foreground">{icon}</div>}
            </div>

            <div className="flex items-end gap-2">
                <span className="text-2xl font-bold tracking-tight text-white">{value}</span>
            </div>

            {subValue && (
                <div className={cn(
                    "text-xs mt-1 font-medium",
                    trend === 'up' ? 'text-green-400' : trend === 'down' ? 'text-red-400' : 'text-muted-foreground'
                )}>
                    {subValue}
                </div>
            )}
        </motion.div>
    );
}
