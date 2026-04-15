"use client";

import { useState, useTransition } from "react";
import { RefreshCw, Wifi, Server, Check } from "lucide-react";
import { useAuthContext } from "@/components/auth/AuthProvider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type BrokerMode = "alpaca" | "ibkr" | "both";

interface Props {
  /** Active broker mode fed from the WS state_update payload. */
  brokerMode: BrokerMode | string;
}

const MODES = [
  {
    id: "alpaca" as BrokerMode,
    label: "Alpaca Native",
    subtitle: "Crypto & Equities via Alpaca only.",
    detail: "Best for paper testing. No TWS required.",
    icon: Wifi,
  },
  {
    id: "both" as BrokerMode,
    label: "IBKR Pro Split",
    subtitle: "IBKR for Equities, Alpaca for Crypto.",
    detail: "Requires TWS running on port 7497.",
    icon: Server,
  },
] as const;

function normalizeMode(raw: string): BrokerMode {
  if (raw === "alpaca" || raw === "ibkr" || raw === "both") return raw;
  return "both";
}

export default function BrokerModePanel({ brokerMode }: Props) {
  const { getToken } = useAuthContext();
  const activeMode = normalizeMode(brokerMode);

  // Local selection — only committed when "Apply" is clicked
  const [selected, setSelected] = useState<BrokerMode>(activeMode);
  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  async function handleApply() {
    if (selected === activeMode) return;
    setError(null);
    setSuccess(false);

    startTransition(async () => {
      try {
        const res = await fetch("/api/v1/broker-mode", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(getToken() ? { authorization: `Bearer ${getToken()}` } : {}),
          },
          body: JSON.stringify({ target_mode: selected }),
          cache: "no-store",
        });
        const data = await res.json().catch(() => ({})) as Record<string, unknown>;
        if (!res.ok) {
          setError(String(data.detail ?? data.error ?? `Request failed (${res.status})`));
        } else {
          setSuccess(true);
          setTimeout(() => setSuccess(false), 3000);
        }
      } catch {
        setError("Network error — backend unreachable.");
      }
    });
  }

  const isDirty = selected !== activeMode;

  return (
    <div className="glass-card rounded-2xl p-5 animate-in fade-in duration-500">
      <div className="mb-5 flex items-center justify-between">
        <div className="flex items-center gap-2">
           <Server className="h-4 w-4 text-primary" />
           <h3 className="text-sm font-bold text-foreground">Multi-Broker Routing</h3>
        </div>
        <Badge variant="secondary" className="bg-background/50 text-[10px] h-6 px-3">
          ACTIVE: {activeMode === "both" ? "IBKR SPLIT" : activeMode.toUpperCase()}
        </Badge>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {MODES.map((mode) => {
          const Icon = mode.icon;
          const isActive = activeMode === mode.id;
          const isSelected = selected === mode.id;
          return (
            <button
              key={mode.id}
              type="button"
              onClick={() => { setSelected(mode.id); setError(null); setSuccess(false); }}
              className={cn(
                "group relative rounded-2xl border p-4 text-left transition-all duration-300",
                isSelected
                  ? "border-primary bg-primary/[0.03] ring-1 ring-primary/40 shadow-lg shadow-primary/5"
                  : "border-border/40 bg-background/20 hover:border-border/80 hover:bg-background/40"
              )}
            >
              <div className="flex items-center gap-3 mb-3">
                <div className={cn(
                  "p-2 rounded-xl transition-colors",
                  isSelected ? "bg-primary/20 text-primary" : "bg-muted/20 text-muted-foreground"
                )}>
                  <Icon className="h-5 w-5" />
                </div>
                <div className="flex-1">
                   <div className="flex items-center gap-2">
                     <span className={cn("text-sm font-bold", isSelected ? "text-primary" : "text-foreground")}>
                       {mode.label}
                     </span>
                     {isActive && (
                       <Badge variant="positive" className="text-[9px] h-4 px-1.5 animate-pulse">
                         LIVE
                       </Badge>
                     )}
                   </div>
                   <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-tight">{mode.subtitle}</p>
                </div>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed pl-12">{mode.detail}</p>
              
              {isSelected && !isActive && (
                <div className="absolute top-4 right-4 text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                  <Check className="h-4 w-4" />
                </div>
              )}
            </button>
          );
        })}
      </div>

      <div className="mt-5 flex items-center gap-4 border-t border-border/40 pt-5">
        <Button
          disabled={!isDirty || isPending}
          onClick={() => void handleApply()}
          className="h-10 px-8 font-bold rounded-xl"
        >
          {isPending && <RefreshCw className="h-4 w-4 animate-spin mr-2" />}
          {isPending ? "Switching..." : "Apply Mode Change"}
        </Button>

        {success && (
          <div className="flex items-center gap-2 text-xs font-bold text-positive animate-in fade-in slide-in-from-left-4">
            <Check className="h-4 w-4" />
            <span>Mode updated — Syncing in 5s</span>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-xs font-bold text-negative animate-in shake duration-300">
            <Server className="h-4 w-4" />
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
}
