"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";

interface ModelVersion {
  version_id: string;
  model_name: string;
  registered_at: number;
  metrics: Record<string, number>;
  status: string;
  promoted_at?: number;
  retired_at?: number;
  notes?: string;
}

interface ModelSummary {
  champion_id?: string;
  champion_ic?: number;
  champion_sharpe?: number;
  champion_promoted_at?: number;
  total_versions: number;
  versions: ModelVersion[];
}

interface RegistryEvent {
  ts: number;
  version_id: string;
  model_name: string;
  action: string;
  reason: string;
  champion_ic_before: number;
  champion_ic_after: number;
}

interface ModelRegistryData {
  available: boolean;
  note?: string;
  total_models?: number;
  total_versions?: number;
  models?: Record<string, ModelSummary>;
  recent_events?: RegistryEvent[];
  ic_promote_delta?: number;
  ic_rollback_thresh?: number;
  sharpe_min?: number;
}

const STATUS_STYLES: Record<string, string> = {
  champion:     "text-green-400 bg-green-400/10 border-green-400/30",
  challenger:   "text-blue-400 bg-blue-400/10 border-blue-400/30",
  retired:      "text-muted-foreground bg-secondary/20 border-border/30",
  rolled_back:  "text-orange-400 bg-orange-400/10 border-orange-400/30",
};

const ACTION_STYLES: Record<string, string> = {
  promote:    "text-green-400",
  rollback:   "text-orange-400",
  register:   "text-blue-400",
};

function StatusBadge({ status }: { status: string }) {
  const cls = STATUS_STYLES[status] ?? "text-muted-foreground bg-secondary/20 border-border/30";
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${cls}`}>
      {status.replace("_", " ")}
    </span>
  );
}

function fmtTs(epoch?: number | null): string {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toISOString().slice(0, 16) + " UTC";
}

export default function ModelRegistryPanel() {
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<ModelRegistryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch("/api/v1/model-registry", {
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
    const id = setInterval(load, 60_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [token]);

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading model registry…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "Model registry unavailable — engine not running."}
      </div>
    );
  }

  const models = Object.entries(data.models ?? {});
  const events = data.recent_events ?? [];

  return (
    <div className="p-4 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground">ML Model Registry</h2>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            {data.total_models ?? 0} models · {data.total_versions ?? 0} total versions · champion/challenger tracking
          </p>
        </div>
        <div className="text-right text-[10px] font-mono text-muted-foreground space-y-0.5">
          <div>IC Δ promote: +{((data.ic_promote_delta ?? 0.005) * 100).toFixed(2)}%</div>
          <div>Sharpe min: {(data.sharpe_min ?? 0.2).toFixed(2)}</div>
        </div>
      </div>

      {/* Models */}
      <div className="space-y-3">
        {models.map(([name, summary]) => (
          <div key={name} className="rounded-xl border border-border/60 bg-background/50">
            <button
              className="w-full p-4 text-left"
              onClick={() => setExpanded(expanded === name ? null : name)}
            >
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-foreground font-mono">{name}</p>
                  {summary.champion_id && (
                    <p className="text-[11px] text-muted-foreground font-mono truncate max-w-[200px]">
                      {summary.champion_id}
                    </p>
                  )}
                </div>
                <div className="text-right space-y-1">
                  <div className="flex items-center gap-2 justify-end">
                    <StatusBadge status={summary.champion_id ? "champion" : "challenger"} />
                    {summary.champion_ic != null && (
                      <span className="text-[11px] font-mono font-semibold text-foreground">
                        IC {(summary.champion_ic * 100).toFixed(2)}%
                      </span>
                    )}
                  </div>
                  <p className="text-[10px] text-muted-foreground font-mono">
                    {summary.total_versions} versions · since {fmtTs(summary.champion_promoted_at)}
                  </p>
                </div>
              </div>
            </button>

            {expanded === name && summary.versions.length > 0 && (
              <div className="border-t border-border/40 px-4 pb-4 pt-3 space-y-1">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">Version History</p>
                <div className="grid grid-cols-5 text-[10px] text-muted-foreground pb-1 border-b border-border/30">
                  <span className="col-span-2">Version</span>
                  <span className="text-right">IC</span>
                  <span className="text-right">Sharpe</span>
                  <span className="text-right">Status</span>
                </div>
                {summary.versions.map((v) => (
                  <div key={v.version_id} className="grid grid-cols-5 text-[11px] py-0.5">
                    <span className="col-span-2 font-mono text-muted-foreground truncate text-[10px]">
                      {new Date((v.registered_at ?? 0) * 1000).toISOString().slice(0, 16)}
                    </span>
                    <span className="text-right font-mono font-semibold text-foreground">
                      {((v.metrics.ic ?? 0) * 100).toFixed(2)}%
                    </span>
                    <span className="text-right font-mono text-muted-foreground">
                      {(v.metrics.sharpe ?? 0).toFixed(2)}
                    </span>
                    <span className="flex justify-end">
                      <StatusBadge status={v.status} />
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {models.length === 0 && (
          <div className="flex items-center justify-center rounded-lg border border-border/40 bg-muted/5 p-8">
            <p className="text-sm text-muted-foreground">No models registered yet — triggers after first weekly retrain.</p>
          </div>
        )}
      </div>

      {/* Event log */}
      {events.length > 0 && (
        <section>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Recent Events
          </h3>
          <div className="space-y-1.5">
            {events.slice(0, 8).map((e, i) => (
              <div key={i} className="flex items-start gap-3 rounded-lg border border-border/30 bg-background/30 px-3 py-2">
                <span className={`text-[11px] font-semibold uppercase w-16 shrink-0 ${ACTION_STYLES[e.action] ?? "text-muted-foreground"}`}>
                  {e.action}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-[11px] text-foreground truncate font-mono">{e.version_id}</p>
                  <p className="text-[10px] text-muted-foreground truncate">{e.reason}</p>
                </div>
                <div className="text-right shrink-0">
                  <p className="text-[10px] font-mono text-muted-foreground">{fmtTs(e.ts)}</p>
                  <p className="text-[10px] font-mono text-muted-foreground">
                    IC {(e.champion_ic_before * 100).toFixed(2)}% → {(e.champion_ic_after * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
