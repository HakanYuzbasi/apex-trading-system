"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/lib/auth-context";

interface VariantState {
  name: string;
  weights: Record<string, number>;
  alpha: number;
  beta_: number;
  n_trades: number;
  win_rate_mean: number;
  created_at?: number;
  promoted_at?: number | null;
}

interface AbGateData {
  available: boolean;
  note?: string;
  control?: VariantState | null;
  challenger?: VariantState | null;
  p_challenger_better?: number | null;
  promotions?: number;
  promotion_history?: Array<{
    ts: number;
    challenger_win_rate: number;
    control_win_rate: number;
    n_challenger_trades: number;
    p_better: number;
  }>;
  thresholds?: {
    min_trades: number;
    promotion_prob: number;
    rollback_drop: number;
    shadow_hours: number;
  };
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function WinRateBar({ value, label }: { value: number; label: string }) {
  const pct = Math.min(Math.max(value, 0), 1) * 100;
  const color = value >= 0.55 ? "bg-green-400" : value >= 0.48 ? "bg-blue-400" : "bg-red-400";
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span>{label}</span>
        <span className="font-mono font-semibold text-foreground">{fmtPct(value)}</span>
      </div>
      <div className="h-2 rounded-full bg-secondary/40 overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${pct.toFixed(1)}%` }} />
      </div>
    </div>
  );
}

function VariantCard({ variant, label, highlight }: { variant: VariantState; label: string; highlight?: boolean }) {
  const entries = Object.entries(variant.weights).sort((a, b) => b[1] - a[1]);
  return (
    <div className={`rounded-xl border p-4 space-y-3 ${highlight ? "border-primary/50 bg-primary/5" : "border-border/60 bg-background/50"}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</p>
          <p className="text-sm font-semibold text-foreground mt-0.5">{variant.name}</p>
        </div>
        <span className="text-[11px] text-muted-foreground font-mono">{variant.n_trades} trades</span>
      </div>
      <WinRateBar value={variant.win_rate_mean} label="Win Rate" />
      <div>
        <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1.5">Signal Weights</p>
        <div className="flex flex-wrap gap-1.5">
          {entries.map(([k, v]) => (
            <span key={k} className="inline-flex items-center gap-1 rounded border border-border/50 bg-secondary/30 px-2 py-0.5 text-[11px] font-mono">
              {k}
              <span className="text-primary font-semibold">{(v * 100).toFixed(0)}%</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function AbGatePanel() {
  const { token } = useAuthContext();
  const [data, setData] = useState<AbGateData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [regInput, setRegInput] = useState("");
  const [regName, setRegName] = useState("challenger-v2");
  const [regError, setRegError] = useState<string | null>(null);
  const [regSuccess, setRegSuccess] = useState(false);

  async function load() {
    try {
      const res = await fetch("/api/v1/ab-gate", {
        headers: token ? { authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const id = setInterval(load, 30_000);
    return () => clearInterval(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function handleRegister() {
    setRegError(null);
    setRegSuccess(false);
    try {
      const weights = JSON.parse(regInput);
      const res = await fetch("/api/v1/ab-gate", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...(token ? { authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ weights, name: regName }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Registration failed");
      }
      setRegSuccess(true);
      setRegInput("");
      await load();
    } catch (e) {
      setRegError(String(e));
    }
  }

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading A/B gate…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "A/B gate unavailable — engine may not be running."}
      </div>
    );
  }

  const pBetter = data.p_challenger_better;
  const thr = data.thresholds;

  return (
    <div className="p-4 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">Signal A/B Gate</h2>
          <p className="text-[11px] text-muted-foreground mt-0.5">Thompson Sampling — challenger promoted when P(challenger &gt; control) ≥ {fmtPct(thr?.promotion_prob)}</p>
        </div>
        <span className="text-[11px] text-muted-foreground font-mono">{data.promotions ?? 0} promotions</span>
      </div>

      {/* P(challenger better) bar */}
      {pBetter != null && data.challenger && (
        <div className="rounded-xl border border-border/60 bg-background/50 p-4 space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-xs text-muted-foreground">P(challenger &gt; control)</span>
            <span className={`text-lg font-bold font-mono ${pBetter >= (thr?.promotion_prob ?? 0.95) ? "text-green-400" : pBetter >= 0.7 ? "text-yellow-400" : "text-muted-foreground"}`}>
              {fmtPct(pBetter)}
            </span>
          </div>
          <div className="h-2.5 rounded-full bg-secondary/40 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${pBetter >= (thr?.promotion_prob ?? 0.95) ? "bg-green-400" : pBetter >= 0.7 ? "bg-yellow-400" : "bg-blue-400"}`}
              style={{ width: `${(pBetter * 100).toFixed(1)}%` }}
            />
          </div>
          <p className="text-[10px] text-muted-foreground">
            {data.challenger.n_trades} / {thr?.min_trades ?? 30} trades · {thr?.shadow_hours ?? 48}h shadow period required
          </p>
        </div>
      )}

      {/* Variant cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {data.control && <VariantCard variant={data.control} label="Control (Live)" />}
        {data.challenger ? (
          <VariantCard variant={data.challenger} label="Challenger (Testing)" highlight />
        ) : (
          <div className="rounded-xl border border-dashed border-border/40 p-4 flex items-center justify-center">
            <p className="text-sm text-muted-foreground">No challenger registered</p>
          </div>
        )}
      </div>

      {/* Promotion history */}
      {(data.promotion_history?.length ?? 0) > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Recent Promotions</h3>
          <div className="space-y-2">
            {data.promotion_history!.map((p, i) => (
              <div key={i} className="flex items-center justify-between rounded-lg border border-border/40 bg-background/50 px-3 py-2 text-xs">
                <span className="text-muted-foreground font-mono">{new Date(p.ts * 1000).toISOString().slice(0, 16)} UTC</span>
                <span className="text-green-400 font-semibold">{fmtPct(p.challenger_win_rate)} WR</span>
                <span className="text-muted-foreground">{p.n_challenger_trades} trades</span>
                <span className="text-blue-400">P={fmtPct(p.p_better)}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Register challenger form */}
      <section className="rounded-xl border border-border/60 bg-background/50 p-4 space-y-3">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Register New Challenger</h3>
        <div className="space-y-2">
          <input
            type="text"
            value={regName}
            onChange={(e) => setRegName(e.target.value)}
            placeholder="Variant name"
            className="w-full rounded-lg border border-border/60 bg-background px-3 py-1.5 text-xs font-mono text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/60"
          />
          <textarea
            value={regInput}
            onChange={(e) => setRegInput(e.target.value)}
            placeholder={'{"ml": 0.60, "tech": 0.25, "sentiment": 0.15}'}
            rows={3}
            className="w-full rounded-lg border border-border/60 bg-background px-3 py-1.5 text-xs font-mono text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/60 resize-none"
          />
        </div>
        {regError && <p className="text-xs text-red-400">{regError}</p>}
        {regSuccess && <p className="text-xs text-green-400">Challenger registered — Thompson Sampling active.</p>}
        <button
          onClick={handleRegister}
          disabled={!regInput.trim()}
          className="rounded-lg bg-primary px-4 py-1.5 text-xs font-semibold text-primary-foreground disabled:opacity-40 hover:bg-primary/90 transition-colors"
        >
          Register Challenger
        </button>
      </section>
    </div>
  );
}
