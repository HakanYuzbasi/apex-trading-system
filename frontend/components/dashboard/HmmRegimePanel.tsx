"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/lib/auth-context";

interface HmmRegimeData {
  available: boolean;
  note?: string;
  method?: string;
  current_label?: string;
  confidence?: number;
  state_probs?: Record<string, number>;
  viterbi_path?: string[];
  trained_at?: number;
  n_train_samples?: number;
  n_states?: number;
  state_map?: Record<string, string>;
  current_vix_regime?: string;
}

const STATE_COLORS: Record<string, string> = {
  bull:     "text-green-400 bg-green-400/10 border-green-400/30",
  neutral:  "text-blue-400 bg-blue-400/10 border-blue-400/30",
  bear:     "text-red-400 bg-red-400/10 border-red-400/30",
  volatile: "text-yellow-400 bg-yellow-400/10 border-yellow-400/30",
};

const STATE_BAR_COLORS: Record<string, string> = {
  bull:     "bg-green-400",
  neutral:  "bg-blue-400",
  bear:     "bg-red-400",
  volatile: "bg-yellow-400",
};

function StateBadge({ label }: { label: string }) {
  const color = STATE_COLORS[label.toLowerCase()] ?? "text-muted-foreground bg-secondary/40 border-border/50";
  return (
    <span className={`inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${color}`}>
      {label}
    </span>
  );
}

function ProbBar({ label, prob }: { label: string; prob: number }) {
  const barColor = STATE_BAR_COLORS[label.toLowerCase()] ?? "bg-muted-foreground/40";
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[11px]">
        <span className="capitalize text-muted-foreground">{label}</span>
        <span className="font-mono font-semibold text-foreground">{(prob * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full bg-secondary/40 overflow-hidden">
        <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${(prob * 100).toFixed(1)}%` }} />
      </div>
    </div>
  );
}

function fmtTs(epoch?: number): string {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toISOString().slice(0, 16) + " UTC";
}

export default function HmmRegimePanel() {
  const { token } = useAuthContext();
  const [data, setData] = useState<HmmRegimeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch("/api/v1/hmm-regime", {
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
    const id = setInterval(load, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [token]);

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading HMM regime…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "HMM Regime unavailable — engine needs 60+ SPY bars to train."}
      </div>
    );
  }

  const label = data.current_label ?? "unknown";
  const conf = data.confidence ?? 0;
  const probs = data.state_probs ?? {};
  const path = data.viterbi_path ?? [];
  const stateOrder = ["bull", "neutral", "bear", "volatile"];

  return (
    <div className="p-4 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">HMM Regime Detector</h2>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            {data.n_states ?? 4}-state Gaussian HMM · trained on {data.n_train_samples ?? 0} samples
          </p>
        </div>
        <span className="text-[10px] text-muted-foreground font-mono">{fmtTs(data.trained_at)}</span>
      </div>

      {/* Current state */}
      <div className="rounded-xl border border-border/60 bg-background/50 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Current HMM State</p>
            <StateBadge label={label} />
          </div>
          <div className="text-right space-y-1">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Confidence</p>
            <p className={`text-2xl font-bold font-mono ${conf >= 0.8 ? "text-green-400" : conf >= 0.6 ? "text-yellow-400" : "text-muted-foreground"}`}>
              {(conf * 100).toFixed(0)}%
            </p>
          </div>
        </div>
        {data.current_vix_regime && (
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
            <span>VIX regime:</span>
            <StateBadge label={data.current_vix_regime} />
          </div>
        )}
      </div>

      {/* State probabilities */}
      {Object.keys(probs).length > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">State Posteriors</h3>
          <div className="space-y-2">
            {stateOrder.filter(s => probs[s] != null).map(s => (
              <ProbBar key={s} label={s} prob={probs[s]} />
            ))}
          </div>
        </section>
      )}

      {/* Viterbi path */}
      {path.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Recent Path (last {path.length} bars)</h3>
          <div className="flex flex-wrap gap-1">
            {path.map((s, i) => (
              <span
                key={i}
                className={`rounded px-1.5 py-0.5 text-[10px] font-mono font-semibold capitalize
                  ${STATE_COLORS[s.toLowerCase()] ?? "text-muted-foreground bg-secondary/30 border-border/40"} border`}
              >
                {s}
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Method badge */}
      <p className="text-[10px] text-muted-foreground">
        Method: <span className="font-mono">{data.method}</span>
        {data.method === "vix_fallback" && (
          <span className="ml-2 text-yellow-400">— install hmmlearn for learned states</span>
        )}
      </p>
    </div>
  );
}
