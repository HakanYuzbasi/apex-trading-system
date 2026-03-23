"use client";

import type {
  ReplayGovernorPolicySnapshot,
  ReplayInspectionResponse,
  ReplayLiquidationProgress,
  ReplayTimelineEvent,
} from "@/lib/api";
import { Button } from "@/components/ui/button";

function formatTs(value?: string | null): string {
  if (!value) return "n/a";
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) return value;
  return new Date(parsed).toLocaleString();
}

function summarizePayload(event: ReplayTimelineEvent): string {
  const payload = event.payload ?? {};
  if (event.event_type === "RISK_DECISION") {
    return `${String(payload.decision ?? "unknown").toUpperCase()} · ${String(payload.stage ?? "stage")} · ${String(payload.reason ?? "reason")}`;
  }
  if (event.event_type === "ORDER_EXECUTION") {
    return `${String(payload.order_role ?? "order")} · ${String(payload.lifecycle ?? "event")} · ${String(payload.status ?? "pending")}`;
  }
  if (event.event_type === "POSITION_UPDATE") {
    return `${String(payload.reason ?? "update")} · qty ${String(payload.quantity ?? "0")}`;
  }
  if (event.event_type === "SIGNAL_GENERATION") {
    return `signal ${Number(payload.signal ?? 0).toFixed(3)} · conf ${Number(payload.confidence ?? 0).toFixed(2)}`;
  }
  if (event.event_type === "STRESS_EVALUATION") {
    return `${String(payload.action ?? "stress")} · ${String(payload.worst_scenario_name ?? payload.worst_scenario_id ?? "scenario")} · return ${formatPercent(payload.worst_portfolio_return)}`;
  }
  if (event.event_type === "STRESS_ACTION") {
    const candidates = Array.isArray(payload.candidates) ? payload.candidates.length : 0;
    const planId = String(payload.liquidation_plan_id ?? "").trim();
    return `${String(payload.action ?? "stress_action")} · ${String(payload.reason ?? "reason")} · ${candidates} candidates${planId ? ` · ${planId}` : ""}`;
  }
  return event.event_type;
}

function formatMetric(value: unknown, digits = 2): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "n/a";
  return numeric.toFixed(digits);
}

function formatPercent(value: unknown, digits = 1): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "n/a";
  return `${(numeric * 100).toFixed(digits)}%`;
}

function liquidationStatusTone(status?: string): string {
  switch (String(status || "").toLowerCase()) {
    case "completed":
      return "text-positive";
    case "in_progress":
      return "text-warning";
    default:
      return "text-muted-foreground";
  }
}

function liquidationSummary(progress: ReplayLiquidationProgress): string {
  return [
    `${progress.status.replaceAll("_", " ")}`,
    `${formatMetric(progress.executed_reduction_qty, 2)} / ${formatMetric(progress.planned_reduction_qty, 2)} cut`,
    `${formatPercent(progress.progress_pct)}`,
  ].join(" · ");
}

function summarizeTierControls(policy: ReplayGovernorPolicySnapshot): string {
  const tier = String(policy.observed_tier ?? "").toLowerCase();
  const controls = tier ? policy.tier_controls?.[tier] : undefined;
  if (!controls) return "No tier controls resolved.";
  return [
    `size x${formatMetric(controls.size_multiplier)}`,
    `signal +${formatMetric(controls.signal_threshold_boost)}`,
    `confidence +${formatMetric(controls.confidence_boost)}`,
    controls.halt_new_entries ? "entries halted" : "entries allowed",
  ].join(" · ");
}

type ReplayInspectorPanelProps = {
  symbol: string | null;
  planId?: string | null;
  targetLabel?: string | null;
  loading: boolean;
  error: string;
  data: ReplayInspectionResponse | null;
  onOpenPlan?: (planId: string) => void;
  onOpenSymbol?: (symbol: string) => void;
  onClose: () => void;
};

export default function ReplayInspectorPanel({
  symbol,
  planId,
  targetLabel,
  loading,
  error,
  data,
  onOpenPlan,
  onOpenSymbol,
  onClose,
}: ReplayInspectorPanelProps) {
  if (!symbol && !planId) return null;

  const isPlanAudit = data?.mode === "plan";
  const headerLabel = targetLabel || (isPlanAudit ? planId : symbol) || "Replay target";
  const latestChain = data?.latest_chain ?? null;
  const timeline = data?.raw_events ?? [];
  const governorPolicy = latestChain?.governor_policy ?? null;
  const liquidation = latestChain?.liquidation_progress ?? null;
  const planAudit = data?.plan_audit ?? null;
  const auditedChains = data?.chains ?? [];

  return (
    <div className="mt-4 rounded-2xl border border-border/80 bg-background/70 p-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Replay Inspector</h3>
          <p className="text-xs text-muted-foreground">
            {headerLabel}
            {data?.summary?.asset_class ? ` · ${data.summary.asset_class}` : ""}
            {data?.summary?.latest_event_at ? ` · latest ${formatTs(data.summary.latest_event_at)}` : ""}
          </p>
        </div>
        <Button type="button" variant="ghost" size="xs" onClick={onClose}>
          Close
        </Button>
      </div>

      {loading ? (
        <p className="mt-3 text-sm text-muted-foreground">Loading replay chain...</p>
      ) : error ? (
        <p className="mt-3 text-sm text-negative">{error}</p>
      ) : !data ? (
        <p className="mt-3 text-sm text-muted-foreground">No replay data loaded.</p>
      ) : (
        <>
          <div className="mt-3 grid gap-2 md:grid-cols-4">
            <div className="rounded-xl border border-border/70 bg-background/60 px-3 py-2">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Chains</p>
              <p className="apex-kpi-value text-lg font-semibold text-foreground">{data.summary.total_chains}</p>
            </div>
            <div className="rounded-xl border border-border/70 bg-background/60 px-3 py-2">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Blocked</p>
              <p className="apex-kpi-value text-lg font-semibold text-negative">{data.summary.blocked_chains}</p>
            </div>
            <div className="rounded-xl border border-border/70 bg-background/60 px-3 py-2">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Filled</p>
              <p className="apex-kpi-value text-lg font-semibold text-positive">{data.summary.filled_chains}</p>
            </div>
            <div className="rounded-xl border border-border/70 bg-background/60 px-3 py-2">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Events</p>
              <p className="apex-kpi-value text-lg font-semibold text-foreground">{data.summary.total_events}</p>
            </div>
            <div className="rounded-xl border border-border/70 bg-background/60 px-3 py-2">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Stress Chains</p>
              <p className="apex-kpi-value text-lg font-semibold text-foreground">{data.summary.stress_liquidation_chains}</p>
            </div>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-[1.15fr_1fr]">
            <div className="rounded-xl border border-border/70 bg-background/60 p-3">
              {isPlanAudit ? (
                <>
                  <div className="flex items-center justify-between gap-3">
                    <h4 className="text-sm font-semibold text-foreground">Liquidation Epoch</h4>
                    {planAudit?.plan_id ? (
                      <span className="rounded-full border border-border/70 px-2 py-0.5 text-[11px] uppercase text-muted-foreground">
                        {planAudit.plan_id}
                      </span>
                    ) : null}
                  </div>
                  {!planAudit ? (
                    <p className="mt-3 text-sm text-muted-foreground">No liquidation plan audit data loaded.</p>
                  ) : (
                    <div className="mt-3 space-y-3 text-xs">
                      <div className="grid gap-2 sm:grid-cols-2">
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Plan ID</p>
                          <p className="font-medium break-all text-foreground">{planAudit.plan_id}</p>
                        </div>
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Epoch</p>
                          <p className="font-medium text-foreground">{planAudit.plan_epoch || "n/a"}</p>
                        </div>
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Started</p>
                          <p className="font-medium text-foreground">{formatTs(planAudit.started_at)}</p>
                        </div>
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Worst scenario</p>
                          <p className="font-medium text-foreground">
                            {planAudit.worst_scenario_name || planAudit.worst_scenario_id || "n/a"}
                          </p>
                        </div>
                      </div>
                      <div className="grid gap-2 sm:grid-cols-3">
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Completed</p>
                          <p className="font-medium text-positive">{planAudit.completed_symbols}</p>
                        </div>
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">In progress</p>
                          <p className="font-medium text-warning">{planAudit.in_progress_symbols}</p>
                        </div>
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Planned</p>
                          <p className="font-medium text-foreground">{planAudit.planned_symbols}</p>
                        </div>
                      </div>
                      <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                        <div className="flex items-center justify-between gap-3">
                          <p className="text-muted-foreground">Symbols in epoch</p>
                          <span className="text-muted-foreground">{planAudit.candidate_symbols.length}</span>
                        </div>
                        <div className="mt-2 space-y-2">
                          {auditedChains.length === 0 ? (
                            <p className="text-muted-foreground">No symbol chains linked to this plan yet.</p>
                          ) : (
                            auditedChains.map((chain) => {
                              const progress = chain.liquidation_progress;
                              return (
                                <div key={`${chain.symbol}-${chain.chain_id}`} className="rounded-lg border border-border/60 bg-background/40 px-3 py-2">
                                  <div className="flex items-center justify-between gap-3">
                                    <div>
                                      <p className="font-medium text-foreground">{chain.symbol}</p>
                                      <p className="text-muted-foreground">
                                        {progress ? liquidationSummary(progress) : chain.final_status}
                                      </p>
                                    </div>
                                    {onOpenSymbol ? (
                                      <Button type="button" size="xs" variant="outline" onClick={() => onOpenSymbol(chain.symbol)}>
                                        Open Symbol
                                      </Button>
                                    ) : null}
                                  </div>
                                  {progress ? (
                                    <div className="mt-2 grid gap-2 sm:grid-cols-3 text-muted-foreground">
                                      <p>Remaining qty {formatMetric(progress.remaining_qty, 2)}</p>
                                      <p>Target {(Number(progress.target_reduction_pct || 0) * 100).toFixed(0)}%</p>
                                      <p>{progress.worst_scenario_name || progress.worst_scenario_id || "scenario"}</p>
                                    </div>
                                  ) : null}
                                </div>
                              );
                            })
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-foreground">Latest Chain</h4>
                    {latestChain ? (
                      <span className="rounded-full border border-border/70 px-2 py-0.5 text-[11px] uppercase text-muted-foreground">
                        {latestChain.final_status}
                      </span>
                    ) : null}
                  </div>
                  {!latestChain ? (
                    <p className="mt-3 text-sm text-muted-foreground">No reconstructed chain for this symbol yet.</p>
                  ) : (
                    <div className="mt-3 space-y-3 text-xs">
                      <div className="grid gap-2 sm:grid-cols-2">
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Chain kind</p>
                          <p className="font-medium text-foreground">{latestChain.chain_kind}</p>
                        </div>
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Terminal reason</p>
                          <p className="font-medium text-foreground">{latestChain.terminal_reason || "n/a"}</p>
                        </div>
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Started</p>
                          <p className="font-medium text-foreground">{formatTs(latestChain.started_at)}</p>
                        </div>
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Completed</p>
                          <p className="font-medium text-foreground">{formatTs(latestChain.completed_at)}</p>
                        </div>
                      </div>

                      {latestChain.signal_event ? (
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Signal snapshot</p>
                          <p className="font-medium text-foreground">{summarizePayload(latestChain.signal_event)}</p>
                        </div>
                      ) : null}

                      <div className="grid gap-2 sm:grid-cols-3">
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Risk decisions</p>
                          <p className="font-medium text-foreground">{latestChain.risk_events.length}</p>
                        </div>
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Order events</p>
                          <p className="font-medium text-foreground">{latestChain.order_events.length}</p>
                        </div>
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <p className="text-muted-foreground">Position events</p>
                          <p className="font-medium text-foreground">{latestChain.position_events.length}</p>
                        </div>
                      </div>

                      {liquidation ? (
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <div className="flex items-center justify-between gap-3">
                            <p className="text-muted-foreground">Liquidation progress</p>
                            <span className={`rounded-full border border-border/70 px-2 py-0.5 text-[11px] uppercase ${liquidationStatusTone(liquidation.status)}`}>
                              {liquidation.status.replaceAll("_", " ")}
                            </span>
                          </div>
                          <div className="mt-2 grid gap-2 sm:grid-cols-2">
                            <div>
                              <p className="text-muted-foreground">Plan</p>
                              <p className="font-medium text-foreground">{liquidationSummary(liquidation)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Worst scenario</p>
                              <p className="font-medium text-foreground">
                                {liquidation.worst_scenario_name || liquidation.worst_scenario_id || "n/a"}
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Remaining qty</p>
                              <p className="font-medium text-foreground">{formatMetric(liquidation.remaining_qty, 2)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Expected stress PnL</p>
                              <p className="font-medium text-foreground">
                                {formatMetric(liquidation.expected_stress_pnl, 0)}
                                {liquidation.remaining_stress_pnl != null
                                  ? ` -> ${formatMetric(liquidation.remaining_stress_pnl, 0)}`
                                  : ""}
                              </p>
                            </div>
                          </div>
                          <div className="mt-2 grid gap-2 sm:grid-cols-[1fr_auto]">
                            <div className="rounded-lg border border-border/60 bg-background/40 px-3 py-2">
                              <p className="text-muted-foreground">Liquidation epoch</p>
                              <p className="font-medium break-all text-foreground">
                                {liquidation.plan_id || "n/a"}
                                {liquidation.plan_epoch > 0 ? ` · epoch ${liquidation.plan_epoch}` : ""}
                              </p>
                            </div>
                            {liquidation.plan_id && onOpenPlan ? (
                              <Button type="button" size="xs" variant="outline" onClick={() => onOpenPlan(liquidation.plan_id)}>
                                Audit Plan
                              </Button>
                            ) : null}
                          </div>
                          <div className="mt-2 h-2 overflow-hidden rounded-full bg-secondary/60">
                            <div
                              className="h-full rounded-full bg-foreground/80 transition-all"
                              style={{ width: `${Math.max(0, Math.min(100, Number(liquidation.progress_pct || 0) * 100))}%` }}
                            />
                          </div>
                          <div className="mt-2 grid gap-2 sm:grid-cols-2">
                            <div className="rounded-lg border border-border/60 bg-background/40 px-3 py-2">
                              <p className="text-muted-foreground">Stress breach</p>
                              <p className="font-medium text-foreground">
                                {liquidation.breach_event ? summarizePayload(liquidation.breach_event) : "n/a"}
                              </p>
                            </div>
                            <div className="rounded-lg border border-border/60 bg-background/40 px-3 py-2">
                              <p className="text-muted-foreground">Liquidation plan</p>
                              <p className="font-medium text-foreground">
                                {liquidation.plan_event ? summarizePayload(liquidation.plan_event) : "n/a"}
                              </p>
                            </div>
                          </div>
                        </div>
                      ) : null}

                      {governorPolicy ? (
                        <div className="rounded-lg border border-border/60 bg-background/50 px-3 py-2">
                          <div className="flex items-center justify-between gap-3">
                            <p className="text-muted-foreground">Governor snapshot</p>
                            <span className="rounded-full border border-border/70 px-2 py-0.5 text-[11px] uppercase text-muted-foreground">
                              {governorPolicy.source || "resolved"}
                            </span>
                          </div>
                          <div className="mt-2 grid gap-2 sm:grid-cols-2">
                            <div>
                              <p className="text-muted-foreground">Policy ID</p>
                              <p className="font-medium text-foreground">{governorPolicy.policy_id}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Version</p>
                              <p className="font-medium text-foreground">{governorPolicy.version}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Regime</p>
                              <p className="font-medium text-foreground">
                                {governorPolicy.asset_class} · {governorPolicy.regime}
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Observed tier</p>
                              <p className="font-medium uppercase text-foreground">
                                {governorPolicy.observed_tier || "n/a"}
                              </p>
                            </div>
                          </div>
                          <div className="mt-2">
                            <p className="text-muted-foreground">Tier behavior</p>
                            <p className="font-medium text-foreground">{summarizeTierControls(governorPolicy)}</p>
                          </div>
                          {governorPolicy.created_at ? (
                            <div className="mt-2">
                              <p className="text-muted-foreground">Created</p>
                              <p className="font-medium text-foreground">{formatTs(governorPolicy.created_at)}</p>
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  )}
                </>
              )}
            </div>

            <div className="rounded-xl border border-border/70 bg-background/60 p-3">
              <h4 className="text-sm font-semibold text-foreground">Timeline</h4>
              <div className="mt-3 max-h-[22rem] space-y-2 overflow-auto pr-1">
                {timeline.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No raw timeline available.</p>
                ) : (
                  timeline.map((event) => (
                    <div key={`${event.hash ?? event.timestamp}-${event.event_type}`} className="rounded-lg border border-border/60 bg-background/50 px-3 py-2 text-xs">
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-medium text-foreground">{event.event_type}</span>
                        <span className="text-muted-foreground">{formatTs(event.timestamp)}</span>
                      </div>
                      <p className="mt-1 text-muted-foreground">{summarizePayload(event)}</p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
