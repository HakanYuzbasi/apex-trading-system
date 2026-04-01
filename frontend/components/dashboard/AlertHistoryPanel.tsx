"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/lib/auth-context";

interface AlertRecord {
  event_type: string;
  message: string;
  ts: number;
  channel: string;
}

interface AlertsData {
  available: boolean;
  note?: string;
  channel?: string;
  alerts?: AlertRecord[];
  total_buffered?: number;
}

const EVENT_STYLES: Record<string, string> = {
  KILL_SWITCH:       "text-red-400 bg-red-400/10 border-red-400/30",
  DRAWDOWN:          "text-orange-400 bg-orange-400/10 border-orange-400/30",
  STRESS_HALT:       "text-red-500 bg-red-500/10 border-red-500/30",
  TRADE_WIN:         "text-green-400 bg-green-400/10 border-green-400/30",
  TRADE_LOSS:        "text-red-400 bg-red-400/10 border-red-400/30",
  MODEL_DRIFT:       "text-yellow-400 bg-yellow-400/10 border-yellow-400/30",
  REGIME_ALERT:      "text-yellow-500 bg-yellow-500/10 border-yellow-500/30",
  EXECUTION_QUALITY: "text-blue-400 bg-blue-400/10 border-blue-400/30",
  EOD_SUMMARY:       "text-blue-400 bg-blue-400/10 border-blue-400/30",
  ENGINE_ERROR:      "text-red-600 bg-red-600/10 border-red-600/30",
};

const EVENT_ICONS: Record<string, string> = {
  KILL_SWITCH:       "🛑",
  DRAWDOWN:          "⚠️",
  STRESS_HALT:       "🚨",
  TRADE_WIN:         "🏆",
  TRADE_LOSS:        "💸",
  MODEL_DRIFT:       "🧠",
  REGIME_ALERT:      "🔴",
  EXECUTION_QUALITY: "⚡",
  EOD_SUMMARY:       "📊",
  ENGINE_ERROR:      "🔴",
};

const CHANNEL_LABELS: Record<string, string> = {
  telegram:         "TG",
  slack:            "SL",
  "telegram+slack": "TG+SL",
  log:              "LOG",
};

function fmtTime(ts: number): string {
  return new Date(ts * 1000).toISOString().slice(11, 19) + " UTC";
}

function AlertBadge({ type }: { type: string }) {
  const style = EVENT_STYLES[type] ?? "text-muted-foreground bg-secondary/30 border-border/40";
  const icon = EVENT_ICONS[type] ?? "•";
  const label = type.replace(/_/g, " ");
  return (
    <span className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${style}`}>
      <span>{icon}</span>{label}
    </span>
  );
}

export default function AlertHistoryPanel() {
  const { token } = useAuthContext();
  const [data, setData] = useState<AlertsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch("/api/v1/alerts?n=100", {
          headers: token ? { authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    const id = setInterval(load, 15_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [token]);

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading alert history…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "Alert history unavailable — engine may not be running."}
      </div>
    );
  }

  const allAlerts = data.alerts ?? [];
  const eventTypes = Array.from(new Set(allAlerts.map((a) => a.event_type)));
  const rows = filter ? allAlerts.filter((a) => a.event_type === filter) : allAlerts;

  const channelLabel = CHANNEL_LABELS[data.channel ?? "log"] ?? data.channel ?? "—";
  const hasLiveChannel = data.channel !== "log_only" && data.channel !== "log";

  return (
    <div className="p-4 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">Alert History</h2>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            Channel:{" "}
            <span className={`font-mono font-semibold ${hasLiveChannel ? "text-green-400" : "text-muted-foreground"}`}>
              {channelLabel}
            </span>
            {!hasLiveChannel && (
              <span className="ml-2 text-yellow-400">— set APEX_TELEGRAM_BOT_TOKEN to enable live push</span>
            )}
          </p>
        </div>
        <span className="text-[11px] text-muted-foreground font-mono">{data.total_buffered ?? 0} total</span>
      </div>

      {/* Filter chips */}
      {eventTypes.length > 1 && (
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setFilter("")}
            className={`rounded-lg border px-3 py-1.5 text-xs transition-colors
              ${filter === "" ? "border-primary/60 bg-primary/10 text-primary" : "border-border/50 text-muted-foreground hover:border-border"}`}
          >
            All <span className="font-mono font-semibold ml-1">{allAlerts.length}</span>
          </button>
          {eventTypes.map((t) => (
            <button
              key={t}
              onClick={() => setFilter(filter === t ? "" : t)}
              className={`rounded-lg border px-3 py-1.5 text-xs transition-colors
                ${filter === t ? "border-primary/60 bg-primary/10 text-primary" : "border-border/50 text-muted-foreground hover:border-border"}`}
            >
              {EVENT_ICONS[t] ?? ""} {t.replace(/_/g, " ").toLowerCase()}
              <span className="font-mono font-semibold ml-1">{allAlerts.filter((a) => a.event_type === t).length}</span>
            </button>
          ))}
        </div>
      )}

      {/* Empty state */}
      {rows.length === 0 && (
        <div className="flex items-center justify-center rounded-lg border border-border/40 bg-green-400/5 p-8">
          <p className="text-sm text-green-400">No alerts{filter ? ` for ${filter}` : ""} — system operating normally.</p>
        </div>
      )}

      {/* Alert feed */}
      {rows.length > 0 && (
        <div className="space-y-2">
          {rows.map((a, i) => (
            <div
              key={i}
              className="rounded-lg border border-border/50 bg-background/50 px-3 py-2.5 space-y-1"
            >
              <div className="flex items-center justify-between gap-2">
                <AlertBadge type={a.event_type} />
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-muted-foreground/60 uppercase">{CHANNEL_LABELS[a.channel] ?? a.channel}</span>
                  <span className="text-[10px] font-mono text-muted-foreground">{fmtTime(a.ts)}</span>
                </div>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed truncate" title={a.message}>
                {a.message}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
