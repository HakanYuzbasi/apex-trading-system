"use client";

import { useState, useTransition } from "react";
import { RefreshCw, Wifi, Server } from "lucide-react";
import { useAuthContext } from "@/components/auth/AuthProvider";

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
    <div className="rounded-2xl border border-border/70 bg-background/60 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Broker Routing</h3>
        <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[11px] font-semibold uppercase text-primary">
          Active: {activeMode === "both" ? "IBKR Split" : activeMode === "alpaca" ? "Alpaca Native" : activeMode.toUpperCase()}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {MODES.map((mode) => {
          const Icon = mode.icon;
          const isActive = activeMode === mode.id;
          const isSelected = selected === mode.id;
          return (
            <button
              key={mode.id}
              type="button"
              onClick={() => { setSelected(mode.id); setError(null); setSuccess(false); }}
              className={[
                "rounded-xl border p-3 text-left transition",
                isSelected
                  ? "border-primary bg-primary/10"
                  : "border-border/70 bg-background/40 hover:bg-secondary/40",
              ].join(" ")}
            >
              <div className="flex items-center gap-2 mb-1">
                <Icon className={`h-4 w-4 ${isSelected ? "text-primary" : "text-muted-foreground"}`} />
                <span className={`text-sm font-semibold ${isSelected ? "text-primary" : "text-foreground"}`}>
                  {mode.label}
                </span>
                {isActive && (
                  <span className="ml-auto rounded-full bg-positive/15 px-1.5 py-0.5 text-[10px] font-semibold text-positive">
                    LIVE
                  </span>
                )}
              </div>
              <p className="text-xs font-medium text-foreground">{mode.subtitle}</p>
              <p className="mt-0.5 text-[11px] text-muted-foreground">{mode.detail}</p>
            </button>
          );
        })}
      </div>

      <div className="mt-3 flex items-center gap-3">
        <button
          type="button"
          disabled={!isDirty || isPending}
          onClick={handleApply}
          className={[
            "inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition",
            isDirty && !isPending
              ? "bg-primary text-primary-foreground hover:bg-primary/90"
              : "cursor-not-allowed bg-secondary text-muted-foreground opacity-50",
          ].join(" ")}
        >
          {isPending && <RefreshCw className="h-3.5 w-3.5 animate-spin" />}
          {isPending ? "Switching…" : "Apply Change"}
        </button>

        {success && (
          <span className="text-xs font-medium text-positive">
            Mode updated — takes effect within 5 s.
          </span>
        )}
      </div>

      {error && (
        <p className="mt-2 rounded-lg border border-negative/30 bg-negative/10 px-3 py-2 text-xs text-negative">
          {error}
        </p>
      )}
    </div>
  );
}
