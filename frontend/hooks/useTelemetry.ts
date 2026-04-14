"use client";

import { useState, useEffect, useCallback, useRef } from 'react';

export interface TelemetryPair {
  pair_name: string;
  z_score: number;
  status: string;
  pnl: number;
}

export interface TelemetryData {
  total_equity: number;
  buying_power: number;
  pairs: TelemetryPair[];
  logs: string[];
  orders: any[];
}

export interface EquityPoint {
  time: string;
  equity: number;
}

const DEFAULT_DATA: TelemetryData = {
  total_equity: 0,
  buying_power: 0,
  pairs: [],
  logs: [],
  orders: [],
};

const MAX_RETRY_DELAY_MS = 30_000;

export function useTelemetry(url: string = "ws://localhost:8765") {
  const [telemetryData, setTelemetryData] = useState<TelemetryData>(DEFAULT_DATA);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(true);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  const retryCountRef = useRef(0);

  const connect = useCallback((retryUrl?: string) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return;

    const targetUrl = retryUrl || url;
    const socketUrl = `${targetUrl}?t=${Date.now()}`;
    console.info(`🛰️ Attempting Telemetry Hookup: ${socketUrl}`);
    setIsConnecting(true);

    const socket = new WebSocket(socketUrl);

    socket.onopen = () => {
      console.log(`Telemetry Bridge Connected ✅ | Endpoint: ${targetUrl}`);
      retryCountRef.current = 0;
      setIsConnected(true);
      setIsConnecting(false);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = undefined;
      }
    };

    socket.onmessage = (event) => {
      try {
        const data: TelemetryData = JSON.parse(event.data);
        setTelemetryData(data);

        if (data.total_equity > 0) {
          setEquityHistory(prev => {
            const newPoint = {
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              equity: data.total_equity
            };
            return [...prev, newPoint].slice(-60);
          });
        }
      } catch (err) {
        console.error("Failed to parse telemetry data:", err);
      }
    };

    socket.onclose = (event) => {
      setIsConnected(false);
      socketRef.current = null;

      if (!event.wasClean) {
        // Fallback: if localhost failed on the first attempt, try 127.0.0.1 immediately
        if (targetUrl.includes("localhost") && retryCountRef.current === 0) {
          const fallbackUrl = targetUrl.replace("localhost", "127.0.0.1");
          console.warn(`Localhost failed. Shifting to fallback: ${fallbackUrl}`);
          connect(fallbackUrl);
        } else {
          retryCountRef.current += 1;
          const delay = Math.min(3000 * Math.pow(1.5, retryCountRef.current - 1), MAX_RETRY_DELAY_MS);
          console.warn(`Telemetry Bridge Connection Lost | Retry #${retryCountRef.current} in ${Math.round(delay / 1000)}s...`);
          // After first failure the socket is gone — mark as not connecting until next attempt fires
          setIsConnecting(false);
          reconnectTimeoutRef.current = setTimeout(() => connect(), delay);
        }
      } else {
        setIsConnecting(false);
      }
    };

    socket.onerror = () => {
      // onerror fires before onclose; onclose handles reconnect logic.
      // The Event object carries no useful detail in the browser — log the URL instead.
      console.warn(`Telemetry WebSocket error on ${targetUrl} (backend may be offline)`);
      socket.close();
    };

    socketRef.current = socket;
  }, [url]);

  const sendEmergencyFlatten = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      console.warn("🚨 EMERGENCY FLATTEN COMMAND SENT");
      socketRef.current.send(JSON.stringify({ command: "EMERGENCY_FLATTEN" }));
    } else {
      console.error("Cannot send Flatten command: WebSocket is not open.");
    }
  }, []);

  const toggleStrategy = useCallback((pairName: string) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      console.info(`🛰️ Uplink: Toggling Strategy [${pairName}]`);
      socketRef.current.send(JSON.stringify({
        command: "TOGGLE_STRATEGY",
        pair_name: pairName
      }));
    } else {
      console.error("Cannot toggle strategy: WebSocket is not open.");
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect]);

  return { isConnected, isConnecting, telemetryData, equityHistory, sendEmergencyFlatten, toggleStrategy };
}
