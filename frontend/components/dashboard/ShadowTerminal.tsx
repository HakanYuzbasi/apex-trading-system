"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Cpu, Wifi, WifiOff } from "lucide-react";
import { type WebSocketMessage } from "@/hooks/useWebSocket";

const SHADOW_MARKER = "[SHADOW MODE] PPO suggests:";

type ShadowTerminalProps = {
  telemetryMessage: WebSocketMessage | null;
  isConnected: boolean;
  isConnecting?: boolean;
  reconnectAttempt?: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function extractShadowLines(message: WebSocketMessage | null): string[] | null {
  if (!message || message.type !== "state_update" || !isRecord(message.shadow_terminal)) {
    return null;
  }

  const lines = Array.isArray(message.shadow_terminal.lines) ? message.shadow_terminal.lines : [];
  const normalized = lines
    .map((line) => String(line ?? "").trim())
    .filter((line) => line.length > 0);

  return normalized.length > 0 ? normalized : [];
}

export default function ShadowTerminal({
  telemetryMessage,
  isConnected,
  isConnecting = false,
  reconnectAttempt = 0,
}: ShadowTerminalProps) {
  const [lines, setLines] = useState<string[]>([]);
  const viewportRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const nextLines = extractShadowLines(telemetryMessage);
    if (nextLines !== null) {
      setLines(nextLines);
    }
  }, [telemetryMessage]);

  useEffect(() => {
    if (!viewportRef.current) {
      return;
    }

    viewportRef.current.scrollTop = viewportRef.current.scrollHeight;
  }, [lines]);

  const connectionLabel = useMemo(() => {
    if (isConnected) {
      return "Shadow feed live";
    }

    if (isConnecting) {
      return "Reconnecting shadow feed";
    }

    if (reconnectAttempt > 0) {
      return `Reconnect attempt ${reconnectAttempt}`;
    }

    return "Shadow feed idle";
  }, [isConnected, isConnecting, reconnectAttempt]);

  return (
    <section className="overflow-hidden rounded-3xl border border-zinc-800/80 bg-zinc-950/95 shadow-[0_24px_60px_rgba(0,0,0,0.45)]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800/80 bg-zinc-900/90 px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-500/20 bg-cyan-500/10">
            <Cpu className="h-4 w-4 text-cyan-300" />
          </div>
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-zinc-300">Shadow Terminal</h2>
            <p className="mt-1 text-xs text-zinc-500">
              Live Neural Execution audit stream logging institutional signals.
            </p>
          </div>
        </div>
        <div className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] ${
          isConnected
            ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
            : "border-amber-500/30 bg-amber-500/10 text-amber-300"
        }`}>
          {isConnected ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
          {connectionLabel}
        </div>
      </div>

      <div
        ref={viewportRef}
        className="h-[280px] overflow-y-auto bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.08),transparent_32%),linear-gradient(180deg,rgba(9,9,11,0.94),rgba(2,6,23,0.98))] px-5 py-4 font-mono text-[12px] leading-6 text-zinc-200"
      >
        {lines.length === 0 ? (
          <div className="flex h-full min-h-[220px] flex-col items-center justify-center gap-3 text-center text-zinc-500">
            <div className="h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_18px_rgba(34,211,238,0.9)]" />
            <p>Waiting for APEX execution events from the live telemetry stream...</p>
          </div>
        ) : (
          <div className="space-y-2">
            {lines.map((line, index) => (
              <div
                key={`${index}-${line.slice(-40)}`}
                className="rounded-xl border border-zinc-800/60 bg-zinc-950/70 px-3 py-2 text-zinc-200"
              >
                <span className="mr-3 text-cyan-300">$</span>
                <span className="break-words whitespace-pre-wrap">{line}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
