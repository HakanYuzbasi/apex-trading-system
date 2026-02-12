"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, XCircle, Info, CheckCircle, X, Bell, BellOff } from "lucide-react";

export type AlertSeverity = "critical" | "warning" | "info" | "success";

export interface AlertItem {
    id: string;
    severity: AlertSeverity;
    title: string;
    message: string;
    timestamp: Date;
    source?: string;
    dismissed?: boolean;
}

interface AlertNotificationsProps {
    alerts: AlertItem[];
    onDismiss: (id: string) => void;
    onDismissAll: () => void;
}

const severityConfig: Record<AlertSeverity, {
    icon: typeof AlertTriangle;
    bgColor: string;
    borderColor: string;
    textColor: string;
    glowColor: string;
}> = {
    critical: {
        icon: XCircle,
        bgColor: "bg-red-950/80",
        borderColor: "border-red-500/50",
        textColor: "text-red-400",
        glowColor: "shadow-[0_0_15px_rgba(239,68,68,0.3)]",
    },
    warning: {
        icon: AlertTriangle,
        bgColor: "bg-yellow-950/80",
        borderColor: "border-yellow-500/50",
        textColor: "text-yellow-400",
        glowColor: "shadow-[0_0_15px_rgba(234,179,8,0.2)]",
    },
    info: {
        icon: Info,
        bgColor: "bg-blue-950/80",
        borderColor: "border-blue-500/50",
        textColor: "text-blue-400",
        glowColor: "",
    },
    success: {
        icon: CheckCircle,
        bgColor: "bg-green-950/80",
        borderColor: "border-green-500/50",
        textColor: "text-green-400",
        glowColor: "",
    },
};

function AlertToast({ alert, onDismiss }: { alert: AlertItem; onDismiss: (id: string) => void }) {
    const config = severityConfig[alert.severity];
    const Icon = config.icon;

    // Auto-dismiss non-critical alerts after 10s
    useEffect(() => {
        if (alert.severity !== "critical") {
            const timer = setTimeout(() => onDismiss(alert.id), 10000);
            return () => clearTimeout(timer);
        }
    }, [alert.id, alert.severity, onDismiss]);

    const timeLabel = alert.timestamp.toLocaleTimeString();

    return (
        <motion.div
            layout
            initial={{ opacity: 0, x: 300, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 300, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
            role={alert.severity === "critical" || alert.severity === "warning" ? "alert" : "status"}
            aria-live={alert.severity === "critical" ? "assertive" : "polite"}
            className={`
                ${config.bgColor} ${config.borderColor} ${config.glowColor}
                border rounded-lg p-3 backdrop-blur-lg max-w-sm w-full
            `}
        >
            <div className="flex items-start gap-3">
                <Icon className={`w-5 h-5 ${config.textColor} shrink-0 mt-0.5`} />
                <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                        <span className={`text-sm font-semibold ${config.textColor}`}>{alert.title}</span>
                        <button
                            onClick={() => onDismiss(alert.id)}
                            className="text-muted-foreground hover:text-white transition-colors shrink-0"
                            aria-label={`Dismiss alert: ${alert.title}`}
                        >
                            <X className="w-3.5 h-3.5" />
                        </button>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{alert.message}</p>
                    <div className="flex items-center gap-2 mt-1.5">
                        {alert.source && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-muted-foreground font-mono">
                                {alert.source}
                            </span>
                        )}
                        <span className="text-[10px] text-muted-foreground/60">{timeLabel}</span>
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

export default function AlertNotifications({ alerts, onDismiss, onDismissAll }: AlertNotificationsProps) {
    const [muted, setMuted] = useState(false);

    const visibleAlerts = alerts.filter(a => !a.dismissed);
    const criticalCount = visibleAlerts.filter(a => a.severity === "critical").length;

    if (muted && criticalCount === 0) {
        return (
            <div className="fixed top-4 right-4 z-50">
                <button
                    onClick={() => setMuted(false)}
                    className="relative glass-panel p-2 rounded-lg hover:bg-white/10 transition-colors"
                    aria-label="Unmute notifications"
                >
                    <BellOff className="w-4 h-4 text-muted-foreground" />
                    {visibleAlerts.length > 0 && (
                        <span className="absolute -top-1 -right-1 w-4 h-4 bg-primary text-[10px] font-bold rounded-full flex items-center justify-center text-black">
                            {visibleAlerts.length}
                        </span>
                    )}
                </button>
            </div>
        );
    }

    return (
        <div className="fixed top-4 right-4 z-50 flex flex-col items-end gap-2 pointer-events-none" aria-live="polite" aria-atomic="false">
            {/* Controls */}
            <div className="flex items-center gap-2 pointer-events-auto">
                {visibleAlerts.length > 1 && (
                    <button
                        onClick={onDismissAll}
                        className="text-[10px] text-muted-foreground hover:text-white transition-colors px-2 py-1 glass-panel rounded"
                        aria-label="Dismiss all alerts"
                    >
                        Clear all
                    </button>
                )}
                <button
                    onClick={() => setMuted(true)}
                    className="glass-panel p-1.5 rounded hover:bg-white/10 transition-colors"
                    title="Mute notifications"
                    aria-label="Mute notifications"
                >
                    <Bell className="w-3.5 h-3.5 text-muted-foreground" />
                </button>
            </div>

            {/* Toast Stack */}
            <div className="flex flex-col gap-2 pointer-events-auto">
                <AnimatePresence mode="popLayout">
                    {visibleAlerts.slice(0, 5).map(alert => (
                        <AlertToast key={alert.id} alert={alert} onDismiss={onDismiss} />
                    ))}
                </AnimatePresence>
                {visibleAlerts.length > 5 && (
                    <div className="text-[10px] text-muted-foreground/60 text-right">
                        +{visibleAlerts.length - 5} more
                    </div>
                )}
            </div>
        </div>
    );
}
