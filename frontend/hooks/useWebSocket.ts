"use client";

import { useRef, useEffect, useState, useCallback } from "react";

export interface WebSocketConfig {
    url: string;
    onMessage?: (data: unknown) => void;
    onConnect?: () => void;
    onDisconnect?: () => void;
    onError?: (error: Event) => void;
    reconnect?: boolean;
    maxReconnectAttempts?: number;
    initialReconnectDelay?: number;
    maxReconnectDelay?: number;
}

export interface WebSocketState {
    isConnected: boolean;
    isConnecting: boolean;
    reconnectAttempt: number;
    lastError: string | null;
    lastConnectedAt: Date | null;
    lastDisconnectedAt: Date | null;
}

export interface UseWebSocketReturn {
    state: WebSocketState;
    send: (data: unknown) => void;
    connect: () => void;
    disconnect: () => void;
}

/**
 * Custom WebSocket hook with exponential backoff reconnection
 */
export function useWebSocket(config: WebSocketConfig): UseWebSocketReturn {
    const {
        url,
        onMessage,
        onConnect,
        onDisconnect,
        onError,
        reconnect = true,
        maxReconnectAttempts = 10,
        initialReconnectDelay = 1000,
        maxReconnectDelay = 30000,
    } = config;

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const reconnectAttemptsRef = useRef(0);
    const isManualDisconnectRef = useRef(false);

    const [state, setState] = useState<WebSocketState>({
        isConnected: false,
        isConnecting: false,
        reconnectAttempt: 0,
        lastError: null,
        lastConnectedAt: null,
        lastDisconnectedAt: null,
    });

    // Calculate reconnect delay with exponential backoff + jitter
    const getReconnectDelay = useCallback(() => {
        const baseDelay = initialReconnectDelay * Math.pow(2, reconnectAttemptsRef.current);
        const jitter = baseDelay * 0.1 * Math.random();
        return Math.min(baseDelay + jitter, maxReconnectDelay);
    }, [initialReconnectDelay, maxReconnectDelay]);

    // Clear reconnect timeout
    const clearReconnectTimeout = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }
    }, []);

    // Schedule reconnection
    const scheduleReconnect = useCallback(() => {
        if (!reconnect || isManualDisconnectRef.current) return;
        if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
            setState(prev => ({
                ...prev,
                lastError: `Max reconnect attempts (${maxReconnectAttempts}) reached`,
            }));
            return;
        }

        const delay = getReconnectDelay();
        console.log(`[WebSocket] Scheduling reconnect in ${Math.round(delay)}ms (attempt ${reconnectAttemptsRef.current + 1}/${maxReconnectAttempts})`);

        reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current += 1;
            setState(prev => ({ ...prev, reconnectAttempt: reconnectAttemptsRef.current }));
            connect();
        }, delay);
    }, [reconnect, maxReconnectAttempts, getReconnectDelay]);

    // Connect function
    const connect = useCallback(() => {
        // Don't connect if already connected or connecting
        if (wsRef.current?.readyState === WebSocket.OPEN) return;
        if (wsRef.current?.readyState === WebSocket.CONNECTING) return;

        clearReconnectTimeout();
        isManualDisconnectRef.current = false;

        setState(prev => ({ ...prev, isConnecting: true, lastError: null }));

        try {
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log("[WebSocket] Connected successfully");
                reconnectAttemptsRef.current = 0;
                setState(prev => ({
                    ...prev,
                    isConnected: true,
                    isConnecting: false,
                    reconnectAttempt: 0,
                    lastConnectedAt: new Date(),
                }));
                onConnect?.();
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    onMessage?.(data);
                } catch {
                    onMessage?.(event.data);
                }
            };

            ws.onerror = (event) => {
                const details = {
                    type: event?.type || "unknown",
                    url,
                    readyState: ws.readyState,
                    reconnectAttempt: reconnectAttemptsRef.current,
                };
                console.warn("[WebSocket] Error:", JSON.stringify(details));
                setState(prev => ({
                    ...prev,
                    lastError: `WebSocket error (${details.type})`,
                }));
                onError?.(event);
            };

            ws.onclose = (event) => {
                console.log(`[WebSocket] Disconnected: code=${event.code}, reason=${event.reason || "N/A"}`);
                wsRef.current = null;
                setState(prev => ({
                    ...prev,
                    isConnected: false,
                    isConnecting: false,
                    lastDisconnectedAt: new Date(),
                }));
                onDisconnect?.();

                // Schedule reconnect if not manual disconnect
                if (!isManualDisconnectRef.current) {
                    scheduleReconnect();
                }
            };
        } catch (error) {
            console.error("[WebSocket] Failed to create connection:", error);
            setState(prev => ({
                ...prev,
                isConnecting: false,
                lastError: error instanceof Error ? error.message : "Failed to connect",
            }));
            scheduleReconnect();
        }
    }, [url, onMessage, onConnect, onDisconnect, onError, clearReconnectTimeout, scheduleReconnect]);

    // Disconnect function
    const disconnect = useCallback(() => {
        isManualDisconnectRef.current = true;
        clearReconnectTimeout();
        reconnectAttemptsRef.current = 0;

        if (wsRef.current) {
            wsRef.current.close(1000, "Manual disconnect");
            wsRef.current = null;
        }

        setState(prev => ({
            ...prev,
            isConnected: false,
            isConnecting: false,
            reconnectAttempt: 0,
        }));
    }, [clearReconnectTimeout]);

    // Send function
    const send = useCallback((data: unknown) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            const message = typeof data === "string" ? data : JSON.stringify(data);
            wsRef.current.send(message);
        } else {
            console.warn("[WebSocket] Cannot send - not connected");
        }
    }, []);

    // Auto-connect on mount (intentionally empty deps - only run once on mount)
    useEffect(() => {
        connect();
        return () => {
            disconnect();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return { state, send, connect, disconnect };
}
