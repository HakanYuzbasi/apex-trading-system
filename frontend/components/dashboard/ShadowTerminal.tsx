"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Cpu, Wifi, WifiOff, Terminal as TerminalIcon } from "lucide-react";
import { type WebSocketMessage } from "@/hooks/useWebSocket";
import { Badge } from "@/components/ui/badge";

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
    if (isConnected) return "Feed Active";
    if (isConnecting) return "Reconnecting...";
    if (reconnectAttempt > 0) return `Attempt ${reconnectAttempt}`;
    return "Feed Idle";
  }, [isConnected, isConnecting, reconnectAttempt]);

  return (
    <section className="glass-card rounded-[2rem] overflow-hidden flex flex-col border border-border/20 shadow-2xl animate-in fade-in zoom-in-95 duration-500">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/10 bg-background/20 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 shadow-[0_0_15px_rgba(var(--primary-rgb),0.1)]">
            <Cpu className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h2 className="text-[11px] font-black uppercase tracking-[0.2em] text-foreground">SHADOW AUDIT LOG</h2>
            <p className="text-[10px] text-muted-foreground/60 font-black uppercase tracking-tighter mt-1 opacity-50">
              Neural Execution stream logging institutional signals.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 rounded-lg border border-primary/20 bg-background/40 px-3 py-1 text-[10px] font-black uppercase tracking-widest text-primary shadow-[0_0_10px_rgba(var(--primary-rgb),0.05)]">
          <div className="relative flex h-1 w-1">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75"></span>
            <span className="relative inline-flex h-1 w-1 rounded-full bg-primary"></span>
          </div>
          {isConnected ? "ACTIVE SYNC" : "DISCONNECTED"}
        </div>
      </div>

      <div
        ref={viewportRef}
        className="flex-1 overflow-y-auto px-6 py-4 custom-scrollbar bg-black/40 font-mono text-[11px] leading-relaxed relative"
      >
        {lines.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 gap-3 opacity-40">
            <p className="text-[10px] font-black uppercase tracking-widest">Awaiting Telemetry</p>
            <p className="text-[10px] font-medium max-w-[200px] text-center">Connecting to the live PPO execution audit stream...</p>
          </div>
        ) : (
          <div className="space-y-2.5">
            {lines.map((line, index) => (
              <div
                key={`${index}-${line.slice(-20)}`}
                className="group relative flex gap-3 rounded-lg border border-border/5 bg-background/10 px-3 py-2 text-foreground/90 transition-all hover:bg-background/20 hover:border-border/10"
              >
                <span className="text-primary font-bold opacity-40 group-hover:opacity-100 transition-opacity">»</span>
                <span className="break-words whitespace-pre-wrap font-medium tracking-tight leading-normal uppercase text-[10px]">{line}</span>
              </div>
            ))}
            <div className="h-4 w-full animate-pulse bg-primary/5 rounded-full" />
          </div>
        )}
        
        {/* Terminal scanline effect */}
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.05)_50%),linear-gradient(90deg,rgba(255,0,0,0.02),rgba(0,255,0,0.01),rgba(0,0,255,0.02))] bg-[length:100%_2px,3px_100%]" />
      </div>
    </section>
  );
}
