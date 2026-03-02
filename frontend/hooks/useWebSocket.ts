"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const STORAGE_ACCESS = "apex_access_token";

interface WebSocketMessage {
    type: string;
    [key: string]: unknown;
}

interface UseWebSocketOptions {
    url?: string;
    onMessage?: (data: WebSocketMessage) => void;
    reconnectInterval?: number;
    shouldConnect?: boolean;
    isPublic?: boolean;
}

export interface WebSocketState {
    isConnected: boolean;
    isConnecting?: boolean;
    reconnectAttempt?: number;
    lastError?: string;
    lastMessage: WebSocketMessage | null;
}

/** Append the stored JWT token as ?token= so the API accepts the WS connection (unless public). */
function getStoredAccessToken(): string | null {
    if (typeof window === "undefined") return null;
    try {
        return localStorage.getItem(STORAGE_ACCESS);
    } catch {
        return null;
    }
}

function buildWsUrl(base: string, isPublic?: boolean): string | null {
    if (typeof window === "undefined") return base;

    let targetBase = base;
    // Redirect localhost:8000/ws to localhost:8000/public/ws if public, though base usually comes from env
    if (isPublic) {
        if (targetBase.endsWith("/ws")) {
            targetBase = targetBase.replace(/\/ws$/, "/public/ws");
        }
        return targetBase; // No token needed
    }

    const token = getStoredAccessToken();
    if (!token) return null;
    const sep = targetBase.includes("?") ? "&" : "?";
    return `${targetBase}${sep}token=${encodeURIComponent(token)}`;
}

const MAX_RECONNECT_ATTEMPTS = Infinity; // Never give up — 24/7 resilience
const MAX_RECONNECT_DELAY_MS = 30_000;

export function useWebSocket(isPublic?: boolean, {
    url = "ws://localhost:8000/ws",
    onMessage,
    reconnectInterval = 3000,
    shouldConnect = true,
}: UseWebSocketOptions = {}) {
    const [state, setState] = useState<WebSocketState>({
        isConnected: false,
        isConnecting: false,
        reconnectAttempt: 0,
        lastError: undefined,
        lastMessage: null,
    });

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
    const attemptRef = useRef(0);
    const stoppedRef = useRef(false); // true when we should NOT reconnect (e.g. 403)
    // Shadow of the last full state — used to reconstruct a complete message from deltas
    const stateShadowRef = useRef<WebSocketMessage | null>(null);
    // Stable ref for onMessage — prevents connect from being recreated when the caller
    // passes an inline arrow function, which would cause the useEffect to re-fire and
    // disconnect/reconnect the socket on every parent render.
    const onMessageRef = useRef(onMessage);
    onMessageRef.current = onMessage;

    const connect = useCallback(function connectSocket() {
        if (!shouldConnect || stoppedRef.current) return;
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        setState((s) => ({ ...s, isConnecting: true }));

        try {
            const baseWsUrl = process.env.NEXT_PUBLIC_WS_URL || url;
            const wsUrl = buildWsUrl(baseWsUrl, isPublic);
            if (!wsUrl) {
                stoppedRef.current = true;
                setState((s) => ({
                    ...s,
                    isConnecting: false,
                    isConnected: false,
                    lastError: "Authentication required for private WebSocket.",
                }));
                return;
            }
            const socket = new WebSocket(wsUrl);

            socket.onopen = () => {
                attemptRef.current = 0;
                stoppedRef.current = false;
                setState((s) => ({
                    ...s,
                    isConnected: true,
                    isConnecting: false,
                    reconnectAttempt: 0,
                    lastError: undefined,
                }));
            };

            socket.onclose = (event) => {
                wsRef.current = null;
                setState((s) => ({ ...s, isConnected: false, isConnecting: false }));

                // 4001/4003 = custom auth codes; 1008 = policy violation (403 maps here)
                const isAuthFailure = event.code === 4001 || event.code === 4003 || event.code === 1008;
                if (isAuthFailure) {
                    stoppedRef.current = true;
                    setState((s) => ({
                        ...s,
                        lastError: "Authentication failed — please log in again.",
                    }));
                    return;
                }

                if (attemptRef.current >= MAX_RECONNECT_ATTEMPTS) {
                    stoppedRef.current = true;
                    setState((s) => ({
                        ...s,
                        lastError: `Gave up after ${MAX_RECONNECT_ATTEMPTS} reconnect attempts.`,
                    }));
                    return;
                }

                // Exponential backoff: 3s, 6s, 12s … capped at 30s
                const delay = Math.min(
                    reconnectInterval * Math.pow(2, attemptRef.current),
                    MAX_RECONNECT_DELAY_MS
                );
                attemptRef.current += 1;
                setState((s) => ({ ...s, reconnectAttempt: attemptRef.current }));
                reconnectTimeoutRef.current = setTimeout(connectSocket, delay);
            };

            socket.onerror = () => {
                // onerror always fires before onclose; let onclose handle reconnect logic.
                // Suppress the noisy "{}" log — the close event has the real info.
                socket.close();
            };

            socket.onmessage = (event) => {
                try {
                    const raw = JSON.parse(event.data as string) as WebSocketMessage;

                    let data: WebSocketMessage;
                    if (raw.type === "state_delta") {
                        // Merge the delta fields into the shadow state and emit as
                        // a full state_update so all consumers remain unchanged.
                        const merged: WebSocketMessage = {
                            ...(stateShadowRef.current ?? {}),
                            ...raw,
                            type: "state_update",
                        };
                        stateShadowRef.current = merged;
                        data = merged;
                    } else {
                        // Full state_update — replace shadow entirely.
                        stateShadowRef.current = raw;
                        data = raw;
                    }

                    setState((s) => ({ ...s, lastMessage: data }));
                    onMessageRef.current?.(data);
                } catch {
                    // Ignore malformed messages
                }
            };

            wsRef.current = socket;
        } catch {
            // Failed to even create the socket (e.g. bad URL)
            const delay = Math.min(
                reconnectInterval * Math.pow(2, attemptRef.current),
                MAX_RECONNECT_DELAY_MS
            );
            attemptRef.current += 1;
            reconnectTimeoutRef.current = setTimeout(connectSocket, delay);
        }
    }, [url, reconnectInterval, shouldConnect, isPublic]);

    useEffect(() => {
        stoppedRef.current = false;
        attemptRef.current = 0;
        connect();

        return () => {
            stoppedRef.current = true;
            if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
            if (wsRef.current) wsRef.current.close();
        };
    }, [connect]);

    // Expose a manual retry so the UI can offer a "Reconnect" button after auth failure
    const retry = useCallback(() => {
        stoppedRef.current = false;
        attemptRef.current = 0;
        setState((s) => ({ ...s, lastError: undefined, reconnectAttempt: 0 }));
        connect();
    }, [connect]);

    return {
        isConnected: state.isConnected,
        isConnecting: state.isConnecting,
        reconnectAttempt: state.reconnectAttempt,
        lastError: state.lastError,
        lastMessage: state.lastMessage,
        retry,
    };
}
