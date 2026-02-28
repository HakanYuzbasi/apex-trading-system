"use client";

import { useEffect, useMemo, useState } from "react";
import type { SocialAuditEvent } from "@/lib/api";

// ─── Helpers ────────────────────────────────────────────────────────────────

function socialDecisionClass(row: SocialAuditEvent): string {
  if (row.block_new_entries) return "bg-negative/15 text-negative";
  if (row.gross_exposure_multiplier < 0.999) return "bg-warning/15 text-warning";
  return "bg-positive/15 text-positive";
}

function socialDecisionLabel(row: SocialAuditEvent): string {
  if (row.block_new_entries) return "BLOCK";
  if (row.gross_exposure_multiplier < 0.999) return `REDUCE ${Math.round(row.gross_exposure_multiplier * 100)}%`;
  return "NORMAL";
}

// ─── Props ──────────────────────────────────────────────────────────────────

export type SocialGovernorPanelProps = {
  socialAudit?: {
    available?: boolean;
    cached?: boolean;
    unauthorized?: boolean;
    transport_error?: boolean;
    count?: number;
    warning?: string | null;
    status_code?: number;
    events?: SocialAuditEvent[];
  };
};

// ─── Component ──────────────────────────────────────────────────────────────

export default function SocialGovernorPanel({ socialAudit }: SocialGovernorPanelProps) {
  const [socialAuditSnapshot, setSocialAuditSnapshot] = useState<SocialAuditEvent[]>([]);

  const socialAuditBaseEvents = useMemo(() => socialAudit?.events ?? [], [socialAudit?.events]);
  const socialAuditFeedDegraded = !socialAudit?.available && !socialAudit?.unauthorized;

  useEffect(() => {
    if (socialAuditBaseEvents.length > 0) {
      setSocialAuditSnapshot(socialAuditBaseEvents);
      return;
    }
    if (!socialAuditFeedDegraded) {
      setSocialAuditSnapshot([]);
    }
  }, [socialAuditBaseEvents, socialAuditFeedDegraded]);

  const socialAuditEvents = useMemo(() => {
    if (socialAuditBaseEvents.length > 0) {
      return socialAuditBaseEvents;
    }
    if (socialAuditFeedDegraded) {
      return socialAuditSnapshot;
    }
    return [];
  }, [socialAuditBaseEvents, socialAuditFeedDegraded, socialAuditSnapshot]);

  const socialAuditIsDegradedHttp = !socialAudit?.available && !socialAudit?.unauthorized && (socialAudit?.status_code ?? 0) >= 500;

  return (
    <section className="grid grid-cols-1 gap-4">
      <article className="apex-panel apex-fade-up rounded-2xl p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold text-foreground">Social Governor Audit</h2>
            <p className="text-xs text-muted-foreground">Immutable decision trail from live social-risk gating.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded-full px-2 py-0.5 text-xs font-semibold uppercase ${socialAudit?.available
              ? socialAudit?.cached
                ? "bg-warning/15 text-warning"
                : "bg-positive/15 text-positive"
              : socialAudit?.unauthorized
                ? "bg-warning/15 text-warning"
                : "bg-negative/15 text-negative"
              }`}>
              {socialAudit?.available ? (socialAudit?.cached ? "cached" : "live") : socialAudit?.unauthorized ? "restricted" : "degraded"}
            </span>
            <span className="rounded-full bg-secondary px-2 py-0.5 text-xs font-semibold text-secondary-foreground">
              {socialAudit?.count ?? 0} rows
            </span>
          </div>
        </div>

        {!socialAudit?.available && socialAudit?.unauthorized ? (
          <p className="rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
            Social-governor audit requires admin scope.
          </p>
        ) : null}

        {!socialAudit?.available && !socialAudit?.unauthorized ? (
          socialAuditIsDegradedHttp ? (
            <p className="mb-2 rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
              Social audit feed is temporarily degraded ({socialAudit?.warning || `http_${socialAudit?.status_code ?? 0}`}). Core cockpit metrics remain available.
            </p>
          ) : (
            <p className="mb-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              Social audit feed unavailable ({socialAudit?.warning || `http_${socialAudit?.status_code ?? 0}`}).
            </p>
          )
        ) : null}
        {socialAudit?.available && socialAudit?.cached ? (
          <p className="mb-2 rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-warning">
            Showing cached social-governor audit snapshot while upstream feed recovers.
          </p>
        ) : null}

        <div className="max-h-[35vh] overflow-auto rounded-xl border border-border/80">
          <table className="min-w-full text-xs">
            <thead className="sticky top-0 z-10 bg-background/95 backdrop-blur">
              <tr className="text-left text-muted-foreground">
                <th className="px-3 py-2 font-semibold">Time</th>
                <th className="px-3 py-2 font-semibold">Scope</th>
                <th className="px-3 py-2 font-semibold">Decision</th>
                <th className="px-3 py-2 font-semibold">Risk</th>
                <th className="px-3 py-2 font-semibold">Verification</th>
                <th className="px-3 py-2 font-semibold">Policy</th>
                <th className="px-3 py-2 font-semibold">Reasons</th>
              </tr>
            </thead>
            <tbody>
              {socialAuditEvents.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-3 py-5 text-center text-muted-foreground">
                    No social-governor audit rows yet.
                  </td>
                </tr>
              ) : (
                socialAuditEvents.slice().reverse().map((row) => (
                  <tr key={row.audit_id} className="border-t border-border/60">
                    <td className="px-3 py-2 text-muted-foreground">
                      {row.timestamp ? new Date(row.timestamp).toLocaleString() : "n/a"}
                    </td>
                    <td className="px-3 py-2 text-foreground">{row.asset_class}/{row.regime}</td>
                    <td className="px-3 py-2">
                      <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${socialDecisionClass(row)}`}>
                        {socialDecisionLabel(row)}
                      </span>
                    </td>
                    <td className="apex-kpi-value px-3 py-2 text-foreground">{(row.combined_risk_score * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 text-muted-foreground">
                      fails {row.prediction_verification_failures} | verified {row.verified_event_count} | p {row.verified_event_probability.toFixed(2)}
                    </td>
                    <td className="px-3 py-2 text-muted-foreground">{row.policy_version || "runtime-config"}</td>
                    <td className="px-3 py-2 text-muted-foreground">
                      {row.reasons.length ? row.reasons.slice(0, 2).join(", ") : "n/a"}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}
