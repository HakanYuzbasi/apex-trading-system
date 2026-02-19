"use client";

import { motion, AnimatePresence } from "framer-motion";
import { WifiOff, RefreshCw, Loader2 } from "lucide-react";
import type { WebSocketState } from "@/hooks/useWebSocket";

interface ConnectionStatusProps {
    wsState: WebSocketState;
    onReconnect: () => void;
}

export default function ConnectionStatus({ wsState, onReconnect }: ConnectionStatusProps) {
    const { isConnected, isConnecting, reconnectAttempt, lastError } = wsState;

    // Show nothing when connected
    if (isConnected) return null;

    return (
        <AnimatePresence>
            <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="shrink-0"
            >
                <div className={`
                    flex items-center justify-between px-4 py-2 rounded-lg text-sm
                    ${isConnecting
                        ? "bg-yellow-950/50 border border-yellow-500/30 text-yellow-400"
                        : "bg-red-950/50 border border-red-500/30 text-red-400"
                    }
                `} role="status" aria-live="polite">
                    <div className="flex items-center gap-3">
                        {isConnecting ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <WifiOff className="w-4 h-4" />
                        )}

                        <span className="font-medium">
                            {isConnecting
                                ? "Reconnecting to APEX..."
                                : "Disconnected from APEX"
                            }
                        </span>

                        {reconnectAttempt !== undefined && reconnectAttempt > 0 && (
                            <span className="text-xs opacity-70 font-mono">
                                Attempt {reconnectAttempt}/10
                            </span>
                        )}

                        {lastError && (
                            <span className="text-xs opacity-60 hidden md:inline">
                                {lastError}
                            </span>
                        )}
                    </div>

                    {!isConnecting && (
                        <button
                            onClick={onReconnect}
                            className="flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium
                                bg-white/10 hover:bg-white/20 transition-colors"
                            aria-label="Reconnect websocket"
                        >
                            <RefreshCw className="w-3 h-3" />
                            Reconnect
                        </button>
                    )}
                </div>
            </motion.div>
        </AnimatePresence>
    );
}
