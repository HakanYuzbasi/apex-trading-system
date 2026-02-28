"use client";

import { useCallback, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatSleeveLabel } from "@/lib/formatters";

// ─── Types ──────────────────────────────────────────────────────────────────

type MandatePolicy = {
  sleeve_allocations: Record<string, number>;
  risk_limits: Record<string, number>;
  execution_constraints: Record<string, number>;
  constraints: string[];
  notes: string[];
};

type MandateEvaluationResult = {
  request_id: string;
  recommendation_mode: "POLICY_ONLY";
  feasible: boolean;
  feasibility_band: "green" | "yellow" | "red";
  probability_target_hit: number;
  confidence: number;
  expected_max_drawdown_pct: number;
  expected_cvar95_pct: number;
  expected_sortino: number;
  rationale: string[];
  warnings: string[];
  policy: MandatePolicy;
  stress_narratives: Array<{
    scenario: string;
    quant_shock: string;
    projected_impact_pct: number;
    mitigation: string;
  }>;
  confidence_intervals_by_regime: Array<{
    regime: string;
    lower: number;
    upper: number;
  }>;
  parsed_mandate?: {
    target_return_pct?: number;
    horizon_days?: number;
    sectors?: string[];
    asset_classes?: string[];
    max_drawdown_pct?: number;
    suitability_profile?: string;
  };
  disclaimer: string;
};

type MandateAuditEvent = {
  timestamp: string;
  model_version: string;
  output_hash: string;
  request: {
    intent?: string;
    sectors?: string[];
  };
  response_summary: {
    request_id?: string;
    feasible?: boolean;
    feasibility_band?: "green" | "yellow" | "red";
    probability_target_hit?: number;
    confidence?: number;
    expected_max_drawdown_pct?: number;
  };
};

type SleeveCalibrationRow = {
  sleeve: string;
  predictions: number;
  predicted_hit_rate: number;
  realized_hit_rate: number;
  calibration_gap: number;
  threshold_abs_gap: number;
  within_threshold: boolean;
  data_quality: string;
};

type MandateCalibrationSnapshot = {
  generated_at: string;
  lookback_events: number;
  rows: SleeveCalibrationRow[];
  notes: string[];
};

type SuitabilityProfile = "conservative" | "balanced" | "aggressive";

type WorkflowSignoff = {
  role: string;
  approved_by: string;
  approved_at: string;
  note: string;
  approved: boolean;
};

type MandateWorkflowPack = {
  workflow_id: string;
  request_id: string;
  created_at: string;
  created_by: string;
  tier: string;
  status: "draft" | "approved" | "paper_live" | "retired";
  recommendation_mode: "POLICY_ONLY";
  feasibility_band: "green" | "yellow" | "red";
  probability_target_hit: number;
  expected_max_drawdown_pct: number;
  required_signoffs: string[];
  signoffs: Record<string, WorkflowSignoff>;
  execution_enabled: boolean;
  notes: string[];
};

type MonthlyModelRiskReport = {
  month: string;
  generated_at: string;
  drift_rows: Array<{
    sleeve: string;
    mean_abs_gap: number;
    max_abs_gap: number;
    threshold_abs_gap: number;
    breach_count: number;
  }>;
  miss_reasons: Record<string, number>;
  policy_changes: Array<{
    timestamp: string;
    workflow_id: string;
    actor: string;
    from_status: string;
    to_status: string;
    note: string;
  }>;
  notes: string[];
};

// ─── Helpers ────────────────────────────────────────────────────────────────

function mandateBandClass(band: string): string {
  if (band === "green") return "bg-positive/15 text-positive";
  if (band === "yellow") return "bg-warning/15 text-warning";
  return "bg-negative/15 text-negative";
}

function workflowStatusClass(status: MandateWorkflowPack["status"]): string {
  if (status === "paper_live") return "bg-primary/15 text-primary";
  if (status === "approved") return "bg-positive/15 text-positive";
  if (status === "retired") return "bg-muted text-muted-foreground";
  return "bg-warning/15 text-warning";
}

// ─── Props ──────────────────────────────────────────────────────────────────

export type MandateCopilotPanelProps = {
  onSessionExpired: () => void;
};

// ─── Component ──────────────────────────────────────────────────────────────

export default function MandateCopilotPanel({ onSessionExpired }: MandateCopilotPanelProps) {
  // Mandate state
  const [mandateIntent, setMandateIntent] = useState("");
  const [mandateSuitability, setMandateSuitability] = useState<SuitabilityProfile>("balanced");
  const [useProfileDrawdown, setUseProfileDrawdown] = useState(true);
  const [mandateDrawdownCap, setMandateDrawdownCap] = useState(15);
  const [mandateHistory, setMandateHistory] = useState<MandateAuditEvent[]>([]);
  const [mandateHistoryLoading, setMandateHistoryLoading] = useState(false);
  const [mandateHistoryError, setMandateHistoryError] = useState("");
  const [mandateResult, setMandateResult] = useState<MandateEvaluationResult | null>(null);
  const [mandateLoading, setMandateLoading] = useState(false);
  const [mandateError, setMandateError] = useState("");
  const [historyDrawerOpen, setHistoryDrawerOpen] = useState(false);

  // Calibration state
  const [calibration, setCalibration] = useState<MandateCalibrationSnapshot | null>(null);
  const [calibrationLoading, setCalibrationLoading] = useState(false);
  const [calibrationError, setCalibrationError] = useState("");

  // Risk report state
  const [monthlyRiskReport, setMonthlyRiskReport] = useState<MonthlyModelRiskReport | null>(null);
  const [riskReportLoading, setRiskReportLoading] = useState(false);
  const [riskReportError, setRiskReportError] = useState("");

  // Workflow state
  const [workflowList, setWorkflowList] = useState<MandateWorkflowPack[]>([]);
  const [workflowListLoading, setWorkflowListLoading] = useState(false);
  const [workflowPack, setWorkflowPack] = useState<MandateWorkflowPack | null>(null);
  const [workflowLoading, setWorkflowLoading] = useState(false);
  const [workflowError, setWorkflowError] = useState("");

  // ── Fetch helpers ──

  const fetchMandateHistory = useCallback(async () => {
    setMandateHistoryLoading(true);
    setMandateHistoryError("");
    try {
      const response = await fetch("/api/v1/mandate/audit?limit=30", { cache: "no-store" });
      const payload = (await response.json().catch(() => ({}))) as { events?: MandateAuditEvent[]; detail?: string };
      if (response.status === 401) {
        onSessionExpired();
        return;
      }
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to load mandate history.");
      }
      setMandateHistory(Array.isArray(payload.events) ? payload.events : []);
    } catch (error: unknown) {
      setMandateHistoryError(error instanceof Error ? error.message : "Failed to load mandate history.");
    } finally {
      setMandateHistoryLoading(false);
    }
  }, [onSessionExpired]);

  const fetchMandateCalibration = useCallback(async () => {
    setCalibrationLoading(true);
    setCalibrationError("");
    try {
      const response = await fetch("/api/v1/mandate/calibration?limit=250", { cache: "no-store" });
      const payload = (await response.json().catch(() => ({}))) as MandateCalibrationSnapshot & { detail?: string };
      if (response.status === 401) {
        onSessionExpired();
        return;
      }
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to load calibration.");
      }
      setCalibration(payload);
    } catch (error: unknown) {
      setCalibrationError(error instanceof Error ? error.message : "Failed to load calibration.");
    } finally {
      setCalibrationLoading(false);
    }
  }, [onSessionExpired]);

  const fetchMonthlyRiskReport = useCallback(async () => {
    setRiskReportLoading(true);
    setRiskReportError("");
    try {
      const response = await fetch("/api/v1/mandate/reports/monthly?lookback=1000", { cache: "no-store" });
      const payload = (await response.json().catch(() => ({}))) as MonthlyModelRiskReport & { detail?: string };
      if (response.status === 401) {
        onSessionExpired();
        return;
      }
      if (response.status === 403) {
        setRiskReportError("Model risk report is available on paid workflow tiers.");
        return;
      }
      if (!response.ok) {
        throw new Error(payload.detail || "Failed to load monthly model risk report.");
      }
      setMonthlyRiskReport(payload);
    } catch (error: unknown) {
      setRiskReportError(error instanceof Error ? error.message : "Failed to load monthly model risk report.");
    } finally {
      setRiskReportLoading(false);
    }
  }, [onSessionExpired]);

  const fetchWorkflowList = useCallback(async () => {
    setWorkflowListLoading(true);
    setWorkflowError("");
    try {
      const response = await fetch("/api/v1/mandate/workflows?limit=50", { cache: "no-store" });
      const payload = (await response.json().catch(() => ([]))) as MandateWorkflowPack[] | { detail?: string };
      if (response.status === 401) {
        onSessionExpired();
        return;
      }
      if (response.status === 403) {
        setWorkflowList([]);
        return;
      }
      if (!response.ok) {
        throw new Error((payload as { detail?: string }).detail || "Failed to load workflow packs.");
      }
      const rows = Array.isArray(payload) ? payload : [];
      setWorkflowList(rows);
      setWorkflowPack((prev) => {
        if (!prev) return rows[0] ?? null;
        return rows.find((row) => row.workflow_id === prev.workflow_id) ?? prev;
      });
    } catch (error: unknown) {
      setWorkflowError(error instanceof Error ? error.message : "Failed to load workflow packs.");
    } finally {
      setWorkflowListLoading(false);
    }
  }, [onSessionExpired]);

  const initiateWorkflowPack = useCallback(async () => {
    if (!mandateResult?.request_id) return;
    setWorkflowLoading(true);
    setWorkflowError("");
    try {
      const response = await fetch("/api/v1/mandate/workflows/initiate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: mandateResult.request_id,
          note: "Initiated from dashboard mandate copilot card.",
        }),
      });
      const payload = (await response.json().catch(() => ({}))) as MandateWorkflowPack & { detail?: string };
      if (response.status === 401) {
        onSessionExpired();
        return;
      }
      if (response.status === 403) {
        setWorkflowError(
          "Workflow initiation is paywalled (Pro+). Admin should pass automatically; regular users need upgrade.",
        );
        return;
      }
      if (!response.ok) {
        throw new Error(payload.detail || "Workflow initiation failed.");
      }
      setWorkflowPack(payload);
      void fetchWorkflowList();
      void fetchMonthlyRiskReport();
    } catch (error: unknown) {
      setWorkflowError(error instanceof Error ? error.message : "Workflow initiation failed.");
    } finally {
      setWorkflowLoading(false);
    }
  }, [mandateResult?.request_id, onSessionExpired, fetchWorkflowList, fetchMonthlyRiskReport]);

  const updateWorkflowStatus = useCallback(async (workflowId: string, status: MandateWorkflowPack["status"], note: string) => {
    setWorkflowLoading(true);
    setWorkflowError("");
    try {
      const response = await fetch(`/api/v1/mandate/workflows/${encodeURIComponent(workflowId)}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status, note }),
      });
      const payload = (await response.json().catch(() => ({}))) as MandateWorkflowPack & { detail?: string };
      if (response.status === 401) {
        onSessionExpired();
        return;
      }
      if (response.status === 403) {
        setWorkflowError("Workflow actions require Pro tier or admin access.");
        return;
      }
      if (!response.ok) {
        throw new Error(payload.detail || "Workflow status update failed.");
      }
      setWorkflowPack(payload);
      void fetchWorkflowList();
      void fetchMonthlyRiskReport();
    } catch (error: unknown) {
      setWorkflowError(error instanceof Error ? error.message : "Workflow status update failed.");
    } finally {
      setWorkflowLoading(false);
    }
  }, [onSessionExpired, fetchWorkflowList, fetchMonthlyRiskReport]);

  const signoffWorkflow = useCallback(async (workflowId: string, role: "pm" | "compliance") => {
    setWorkflowLoading(true);
    setWorkflowError("");
    try {
      const response = await fetch(`/api/v1/mandate/workflows/${encodeURIComponent(workflowId)}/signoff`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role,
          note: `Sign-off recorded from PM cockpit (${role}).`,
        }),
      });
      const payload = (await response.json().catch(() => ({}))) as MandateWorkflowPack & { detail?: string };
      if (response.status === 401) {
        onSessionExpired();
        return;
      }
      if (response.status === 403) {
        setWorkflowError(role === "compliance"
          ? "Compliance sign-off requires admin/compliance role."
          : "Workflow actions require Pro tier or admin access.");
        return;
      }
      if (!response.ok) {
        throw new Error(payload.detail || "Workflow sign-off failed.");
      }
      setWorkflowPack(payload);
      void fetchWorkflowList();
      void fetchMonthlyRiskReport();
    } catch (error: unknown) {
      setWorkflowError(error instanceof Error ? error.message : "Workflow sign-off failed.");
    } finally {
      setWorkflowLoading(false);
    }
  }, [onSessionExpired, fetchWorkflowList, fetchMonthlyRiskReport]);

  const activatePlanFlow = useCallback(async () => {
    if (!mandateResult?.request_id) {
      setWorkflowError("Run mandate evaluation first.");
      return;
    }
    const parsedMandate = mandateResult.parsed_mandate ?? {};
    const targetReturnPct = Number(parsedMandate.target_return_pct ?? 0);
    const horizonDays = Number(parsedMandate.horizon_days ?? 0);
    const activationContext = Number.isFinite(targetReturnPct) && targetReturnPct > 0 && Number.isFinite(horizonDays) && horizonDays > 0
      ? `target=${targetReturnPct.toFixed(1)}% horizon=${Math.trunc(horizonDays)}d`
      : "target/horizon parsed from mandate intent";

    setWorkflowLoading(true);
    setWorkflowError("");
    try {
      let activePack: MandateWorkflowPack | null = workflowPack && workflowPack.status !== "retired"
        ? workflowPack
        : null;

      if (!activePack || activePack.request_id !== mandateResult.request_id) {
        const initiateResponse = await fetch("/api/v1/mandate/workflows/initiate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            request_id: mandateResult.request_id,
            note: `Auto-initiated from plan activation (${activationContext}).`,
          }),
        });
        const initiatePayload = (await initiateResponse.json().catch(() => ({}))) as MandateWorkflowPack & { detail?: string };
        if (initiateResponse.status === 401) {
          onSessionExpired();
          return;
        }
        if (initiateResponse.status === 403) {
          setWorkflowError("Plan activation requires workflow-pack entitlement (Pro+) or admin role.");
          return;
        }
        if (!initiateResponse.ok) {
          throw new Error(initiatePayload.detail || "Workflow initiation failed.");
        }
        activePack = initiatePayload;
      }

      if (!activePack.signoffs?.pm?.approved) {
        const pmResponse = await fetch(`/api/v1/mandate/workflows/${encodeURIComponent(activePack.workflow_id)}/signoff`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            role: "pm",
            note: `Auto PM sign-off (${activationContext}).`,
          }),
        });
        const pmPayload = (await pmResponse.json().catch(() => ({}))) as MandateWorkflowPack & { detail?: string };
        if (pmResponse.status === 401) {
          onSessionExpired();
          return;
        }
        if (pmResponse.status === 403) {
          setWorkflowError("PM sign-off requires workflow pack access.");
          return;
        }
        if (!pmResponse.ok) {
          throw new Error(pmPayload.detail || "PM sign-off failed.");
        }
        activePack = pmPayload;
      }

      if (!activePack.signoffs?.compliance?.approved) {
        const complianceResponse = await fetch(`/api/v1/mandate/workflows/${encodeURIComponent(activePack.workflow_id)}/signoff`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            role: "compliance",
            note: `Auto compliance sign-off (${activationContext}).`,
          }),
        });
        const compliancePayload = (await complianceResponse.json().catch(() => ({}))) as MandateWorkflowPack & { detail?: string };
        if (complianceResponse.status === 401) {
          onSessionExpired();
          return;
        }
        if (complianceResponse.status === 403) {
          setWorkflowError("Compliance sign-off requires admin/compliance role.");
          return;
        }
        if (!complianceResponse.ok) {
          throw new Error(compliancePayload.detail || "Compliance sign-off failed.");
        }
        activePack = compliancePayload;
      }

      if (activePack.status !== "paper_live") {
        const statusResponse = await fetch(`/api/v1/mandate/workflows/${encodeURIComponent(activePack.workflow_id)}/status`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            status: "paper_live",
            note: `Applied to paper runtime with ${activationContext}.`,
          }),
        });
        const statusPayload = (await statusResponse.json().catch(() => ({}))) as MandateWorkflowPack & { detail?: string };
        if (statusResponse.status === 401) {
          onSessionExpired();
          return;
        }
        if (statusResponse.status === 403) {
          setWorkflowError("Policy activation requires workflow pack access.");
          return;
        }
        if (!statusResponse.ok) {
          throw new Error(statusPayload.detail || "Paper runtime activation failed.");
        }
        activePack = statusPayload;
      }

      setWorkflowPack(activePack);
      void fetchWorkflowList();
      void fetchMonthlyRiskReport();
      void fetchMandateCalibration();
    } catch (error: unknown) {
      setWorkflowError(error instanceof Error ? error.message : "Plan activation failed.");
    } finally {
      setWorkflowLoading(false);
    }
  }, [mandateResult, workflowPack, onSessionExpired, fetchWorkflowList, fetchMonthlyRiskReport, fetchMandateCalibration]);

  const evaluateMandate = useCallback(async () => {
    setMandateLoading(true);
    setMandateError("");
    try {
      const requestBody: Record<string, unknown> = {
        intent: mandateIntent,
        suitability_profile: mandateSuitability,
        recommendation_mode: "POLICY_ONLY",
      };
      if (!useProfileDrawdown) {
        requestBody.max_drawdown_pct = mandateDrawdownCap;
      }
      const response = await fetch("/api/v1/mandate/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      const payload = (await response.json().catch(() => ({}))) as MandateEvaluationResult & { detail?: string };
      if (response.status === 401) {
        onSessionExpired();
        return;
      }
      if (!response.ok) {
        throw new Error(payload.detail || "Mandate evaluation failed.");
      }
      setMandateResult(payload);
      void fetchMandateHistory();
      void fetchMandateCalibration();
      void fetchMonthlyRiskReport();
      void fetchWorkflowList();
    } catch (error: unknown) {
      setMandateError(error instanceof Error ? error.message : "Mandate evaluation failed.");
    } finally {
      setMandateLoading(false);
    }
  }, [mandateIntent, mandateSuitability, useProfileDrawdown, mandateDrawdownCap, onSessionExpired, fetchMandateHistory, fetchMandateCalibration, fetchMonthlyRiskReport, fetchWorkflowList]);

  const pmSigned = Boolean(workflowPack?.signoffs?.pm?.approved);
  const complianceSigned = Boolean(workflowPack?.signoffs?.compliance?.approved);

  // Expose copilot signal count for parent readiness display
  const copilotSignals = useMemo(
    () => (calibration?.rows?.length ?? 0) + mandateHistory.length + (mandateResult ? 1 : 0),
    [calibration?.rows?.length, mandateHistory.length, mandateResult],
  );

  // Memoize the readiness summary so Dashboard can consume it without re-mounting
  const readinessSummary = useMemo(() => ({
    copilotSignals,
    mandateError,
    calibrationError,
    workflowError,
    workflowListLength: workflowList.length,
    workflowPackStatus: workflowPack?.status ?? null,
  }), [copilotSignals, mandateError, calibrationError, workflowError, workflowList.length, workflowPack?.status]);

  // Expose for parent
  void readinessSummary;

  return (
    <section className="grid grid-cols-1 gap-4">
      <article className="apex-panel apex-fade-up rounded-2xl p-5">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-foreground">AI Mandate Copilot (Plan Activation)</h2>
            <p className="text-sm text-muted-foreground">
              Goal-to-policy assessment with DD/CVaR constraints. Approved plans can be applied to paper runtime controls (no live order routing).
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-secondary px-3 py-1 text-xs font-semibold text-secondary-foreground">
              Recommendation Mode: POLICY_ONLY
            </span>
            <Button
              size="sm"
              variant="outline"
              className="h-9 rounded-full px-3 text-xs"
              onClick={() => {
                setHistoryDrawerOpen((prev) => !prev);
                if (!historyDrawerOpen) {
                  void fetchMandateHistory();
                }
              }}
            >
              {historyDrawerOpen ? "Hide History" : "History"}
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-9 rounded-full px-3 text-xs"
              onClick={() => {
                void fetchMandateCalibration();
              }}
              disabled={calibrationLoading}
            >
              {calibrationLoading ? <><RefreshCw className="h-3.5 w-3.5 animate-spin" /> Refreshing...</> : "Refresh Calibration"}
            </Button>
          </div>
        </div>

        <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_auto_auto_auto] lg:items-end">
          <label className="space-y-1 text-sm">
            <span className="text-muted-foreground">Mandate Prompt</span>
            <textarea
              aria-label="Mandate prompt"
              className="min-h-[96px] w-full rounded-xl border border-border/80 bg-background/70 px-3 py-2 text-sm text-foreground outline-none ring-0 placeholder:text-muted-foreground focus:border-primary/60"
              value={mandateIntent}
              onChange={(event) => setMandateIntent(event.target.value)}
              placeholder="Example: I want to make 10% in two months in energy and tech."
            />
          </label>
          <label className="space-y-1 text-sm">
            <span className="text-muted-foreground">Suitability Profile</span>
            <select
              aria-label="Suitability profile"
              className="h-10 w-full rounded-xl border border-border/80 bg-background/70 px-3 text-sm capitalize text-foreground outline-none ring-0 focus:border-primary/60"
              value={mandateSuitability}
              onChange={(event) => setMandateSuitability(event.target.value as SuitabilityProfile)}
            >
              <option value="conservative">Conservative (10% DD)</option>
              <option value="balanced">Balanced (15% DD)</option>
              <option value="aggressive">Aggressive (25% DD)</option>
            </select>
          </label>
          <div className="space-y-2 text-sm">
            <label className="flex items-center gap-2 text-muted-foreground">
              <input
                aria-label="Use profile drawdown default"
                type="checkbox"
                checked={useProfileDrawdown}
                onChange={(event) => setUseProfileDrawdown(event.target.checked)}
              />
              Use profile DD default
            </label>
            <span className="text-xs text-muted-foreground">
              {useProfileDrawdown ? "Profile bound DD applied." : "Manual DD override active."}
            </span>
            <input
              aria-label="Max drawdown percent"
              type="number"
              min={5}
              max={50}
              step={0.5}
              disabled={useProfileDrawdown}
              className="h-10 w-full rounded-xl border border-border/80 bg-background/70 px-3 text-sm text-foreground outline-none ring-0 disabled:cursor-not-allowed disabled:opacity-50 focus:border-primary/60"
              value={mandateDrawdownCap}
              onChange={(event) => setMandateDrawdownCap(Number(event.target.value))}
            />
          </div>
          <Button
            className="h-10 rounded-xl"
            onClick={() => {
              void evaluateMandate();
            }}
            disabled={mandateLoading}
          >
            {mandateLoading ? <><RefreshCw className="h-3.5 w-3.5 animate-spin" /> Evaluating...</> : "Evaluate Mandate"}
          </Button>
        </div>

        {mandateError ? (
          <p className="mt-3 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {mandateError}
          </p>
        ) : null}

        {historyDrawerOpen ? (
          <div className="mt-3 rounded-xl border border-border/80 bg-background/60 p-3">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-sm font-semibold text-foreground">Recent Mandate Assessments</p>
              <span className="text-xs text-muted-foreground">{mandateHistory.length} events</span>
            </div>
            {mandateHistoryError ? (
              <p className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {mandateHistoryError}
              </p>
            ) : null}
            <div className="max-h-[30vh] space-y-2 overflow-auto">
              {mandateHistoryLoading ? (
                <p className="text-xs text-muted-foreground">Loading history...</p>
              ) : mandateHistory.length === 0 ? (
                <p className="text-xs text-muted-foreground">No mandate history yet.</p>
              ) : (
                mandateHistory.map((event) => (
                  <div key={`${event.output_hash}-${event.timestamp}`} className="rounded-lg border border-border/70 bg-background/70 px-3 py-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase ${mandateBandClass(event.response_summary.feasibility_band || "red")}`}>
                        {event.response_summary.feasibility_band || "unknown"}
                      </span>
                      <span className="text-[11px] text-muted-foreground">{new Date(event.timestamp).toLocaleString()}</span>
                      <span className="text-[11px] text-muted-foreground">{event.response_summary.request_id || "n/a"}</span>
                    </div>
                    <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                      {event.request?.intent || "No intent text captured."}
                    </p>
                    <p className="mt-1 text-[11px] text-muted-foreground">
                      P(hit): {(((event.response_summary.probability_target_hit || 0) as number) * 100).toFixed(1)}% | Confidence:{" "}
                      {(((event.response_summary.confidence || 0) as number) * 100).toFixed(1)}% | Max DD:{" "}
                      {Number(event.response_summary.expected_max_drawdown_pct || 0).toFixed(1)}%
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        ) : null}

        {mandateResult ? (
          <div className="mt-4 grid gap-3 lg:grid-cols-4">
            <div className="rounded-xl border border-border/80 bg-background/70 p-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Feasibility</p>
              <div className="mt-2 flex items-center gap-2">
                <span className={`rounded-full px-2 py-0.5 text-xs font-semibold uppercase ${mandateBandClass(mandateResult.feasibility_band)}`}>
                  {mandateResult.feasibility_band}
                </span>
                <span className="text-xs text-muted-foreground">{mandateResult.feasible ? "Feasible" : "Not Feasible"}</span>
              </div>
            </div>
            <div className="rounded-xl border border-border/80 bg-background/70 p-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">P(target hit)</p>
              <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">
                {(mandateResult.probability_target_hit * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground">Confidence {(mandateResult.confidence * 100).toFixed(1)}%</p>
            </div>
            <div className="rounded-xl border border-border/80 bg-background/70 p-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Risk Projection</p>
              <p className="apex-kpi-value mt-2 text-sm font-semibold text-foreground">
                Max DD {mandateResult.expected_max_drawdown_pct.toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground">CVaR95 {mandateResult.expected_cvar95_pct.toFixed(1)}%</p>
            </div>
            <div className="rounded-xl border border-border/80 bg-background/70 p-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Policy Constraints</p>
              <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
                {mandateResult.policy.constraints.slice(0, 3).map((constraint) => (
                  <li key={constraint}>- {constraint}</li>
                ))}
              </ul>
            </div>
            <div className="rounded-xl border border-border/80 bg-background/70 p-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Parsed Mandate</p>
              <p className="apex-kpi-value mt-2 text-sm font-semibold text-foreground">
                Target {(Number(mandateResult.parsed_mandate?.target_return_pct ?? 0) || 0).toFixed(1)}%
              </p>
              <p className="text-xs text-muted-foreground">
                Horizon {Math.trunc(Number(mandateResult.parsed_mandate?.horizon_days ?? 0) || 0)}d • DD cap {(Number(mandateResult.parsed_mandate?.max_drawdown_pct ?? 0) || 0).toFixed(1)}%
              </p>
            </div>

            <div className="rounded-xl border border-border/80 bg-background/70 p-3 lg:col-span-2">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Sleeve Allocation Draft</p>
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                {Object.entries(mandateResult.policy.sleeve_allocations).map(([sleeve, weight]) => (
                  <div key={sleeve} className="rounded-lg border border-border/70 bg-background/60 px-2 py-1.5">
                    <p className="capitalize text-foreground">{formatSleeveLabel(sleeve)}</p>
                    <p className="text-muted-foreground">{(Number(weight) * 100).toFixed(1)}%</p>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-xl border border-border/80 bg-background/70 p-3 lg:col-span-2">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Rationale</p>
              <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
                {mandateResult.rationale.slice(0, 3).map((item) => (
                  <li key={item}>- {item}</li>
                ))}
              </ul>
              <p className="mt-2 text-[11px] text-muted-foreground">{mandateResult.disclaimer}</p>
            </div>

            <div className="rounded-xl border border-border/80 bg-background/70 p-3 lg:col-span-2">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Stress Narratives</p>
              <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
                {mandateResult.stress_narratives?.slice(0, 3).map((row) => (
                  <li key={`${row.scenario}-${row.quant_shock}`}>
                    - {row.scenario}: {row.quant_shock} ({row.projected_impact_pct.toFixed(1)}%) | {row.mitigation}
                  </li>
                ))}
              </ul>
            </div>

            <div className="rounded-xl border border-border/80 bg-background/70 p-3 lg:col-span-2">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Regime Confidence Intervals</p>
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                {(mandateResult.confidence_intervals_by_regime || []).slice(0, 4).map((interval) => (
                  <div key={interval.regime} className="rounded-lg border border-border/70 bg-background/60 px-2 py-1.5">
                    <p className="uppercase text-foreground">{interval.regime}</p>
                    <p className="text-muted-foreground">
                      {(interval.lower * 100).toFixed(1)}% - {(interval.upper * 100).toFixed(1)}%
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <p className="mt-4 rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
            Run an evaluation to view feasibility, confidence, and policy constraints.
          </p>
        )}

        {/* ── Workflow Pack ── */}
        <div className="mt-4 rounded-xl border border-border/80 bg-background/60 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-semibold text-foreground">Advisor Workflow Pack (PM + Compliance)</p>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                className="h-9 rounded-full px-3 text-xs"
                onClick={() => {
                  void initiateWorkflowPack();
                }}
                disabled={workflowLoading || !mandateResult?.request_id}
              >
                {workflowLoading ? "Initiating..." : "Initiate Workflow"}
              </Button>
              <Button
                size="sm"
                className="h-9 rounded-full px-3 text-xs"
                onClick={() => {
                  void activatePlanFlow();
                }}
                disabled={workflowLoading || !mandateResult?.request_id}
              >
                {workflowLoading ? "Activating..." : "Activate Plan (Auto)"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="h-9 rounded-full px-3 text-xs"
                onClick={() => {
                  void fetchWorkflowList();
                }}
                disabled={workflowListLoading}
              >
                {workflowListLoading ? "Refreshing..." : "Refresh"}
              </Button>
            </div>
          </div>
          {workflowError ? (
            <p className="mt-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {workflowError}
            </p>
          ) : null}
          {workflowPack ? (
            <div className="mt-2 space-y-2 rounded-lg border border-border/70 bg-background/70 px-3 py-2 text-xs text-muted-foreground">
              <div className="flex flex-wrap items-center gap-2">
                <p>
                  Workflow <span className="font-semibold text-foreground">{workflowPack.workflow_id}</span>
                </p>
                <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase ${workflowStatusClass(workflowPack.status)}`}>
                  {workflowPack.status}
                </span>
              </div>
              <p>Runtime action: {workflowPack.execution_enabled ? "policy integrity issue detected" : "paper policy activation enabled (no live routing)"}</p>
              <div className="flex flex-wrap items-center gap-2">
                <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${pmSigned ? "bg-positive/15 text-positive" : "bg-warning/15 text-warning"}`}>
                  PM: {pmSigned ? "signed" : "pending"}
                </span>
                <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${complianceSigned ? "bg-positive/15 text-positive" : "bg-warning/15 text-warning"}`}>
                  Compliance: {complianceSigned ? "signed" : "pending"}
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  className="h-9 rounded-full px-3 text-xs"
                  onClick={() => {
                    void signoffWorkflow(workflowPack.workflow_id, "pm");
                  }}
                  disabled={workflowLoading || pmSigned}
                >
                  PM Sign-off
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-9 rounded-full px-3 text-xs"
                  onClick={() => {
                    void signoffWorkflow(workflowPack.workflow_id, "compliance");
                  }}
                  disabled={workflowLoading || complianceSigned}
                >
                  Compliance Sign-off
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-9 rounded-full px-3 text-xs"
                  onClick={() => {
                    void updateWorkflowStatus(
                      workflowPack.workflow_id,
                      "paper_live",
                      "Applied to paper runtime after PM + compliance sign-off.",
                    );
                  }}
                  disabled={workflowLoading || workflowPack.status !== "approved"}
                >
                  Apply to Paper Runtime
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-9 rounded-full px-3 text-xs"
                  onClick={() => {
                    void updateWorkflowStatus(
                      workflowPack.workflow_id,
                      "retired",
                      "Retired from cockpit review.",
                    );
                  }}
                  disabled={workflowLoading || workflowPack.status === "retired"}
                >
                  Retire
                </Button>
              </div>
            </div>
          ) : (
            <p className="mt-2 text-xs text-muted-foreground">
              Workflow packs remain live-order isolated. Approved packs can apply policy controls to paper runtime.
            </p>
          )}

          <div className="mt-3 rounded-lg border border-border/70">
            <div className="flex items-center justify-between border-b border-border/60 px-3 py-2">
              <p className="text-xs font-semibold text-foreground">Workflow History</p>
              <span className="text-[11px] text-muted-foreground">{workflowList.length} packs</span>
            </div>
            <div className="max-h-[25vh] overflow-auto">
              {workflowList.length === 0 ? (
                <p className="px-3 py-3 text-xs text-muted-foreground">No workflow packs yet or paywall restricted.</p>
              ) : (
                workflowList.slice(0, 8).map((row) => (
                  <button
                    key={row.workflow_id}
                    type="button"
                    className="flex w-full items-center justify-between border-t border-border/60 px-3 py-2 text-left text-xs hover:bg-secondary/30"
                    onClick={() => setWorkflowPack(row)}
                  >
                    <span className="text-foreground">{row.workflow_id}</span>
                    <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase ${workflowStatusClass(row.status)}`}>
                      {row.status}
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        {/* ── Calibration Table ── */}
        <div className="mt-4 rounded-xl border border-border/80 bg-background/60 p-3">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-sm font-semibold text-foreground">Calibration (Predicted vs Realized by Sleeve)</p>
            <span className="text-xs text-muted-foreground">
              {calibration?.lookback_events ?? 0} mandate events
            </span>
          </div>
          {calibrationError ? (
            <p className="mb-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {calibrationError}
            </p>
          ) : null}
          <div className="max-h-[30vh] overflow-auto rounded-lg border border-border/70">
            <table className="min-w-full text-xs">
              <thead className="sticky top-0 z-10 bg-background/95">
                <tr className="text-left text-muted-foreground">
                  <th className="px-3 py-2 font-semibold">Sleeve</th>
                  <th className="px-3 py-2 font-semibold">N</th>
                  <th className="px-3 py-2 font-semibold">Predicted</th>
                  <th className="px-3 py-2 font-semibold">Realized</th>
                  <th className="px-3 py-2 font-semibold">Gap</th>
                  <th className="px-3 py-2 font-semibold">Threshold</th>
                  <th className="px-3 py-2 font-semibold">Pass</th>
                  <th className="px-3 py-2 font-semibold">Quality</th>
                </tr>
              </thead>
              <tbody>
                {calibrationLoading ? (
                  <tr>
                    <td colSpan={8} className="px-3 py-5 text-center text-muted-foreground">
                      Loading calibration...
                    </td>
                  </tr>
                ) : (calibration?.rows?.length || 0) === 0 ? (
                  <tr>
                    <td colSpan={8} className="px-3 py-5 text-center text-muted-foreground">
                      No calibration data yet.
                    </td>
                  </tr>
                ) : (
                  (calibration?.rows ?? []).map((row) => (
                    <tr key={row.sleeve} className="border-t border-border/60">
                      <td className="px-3 py-2 capitalize text-foreground">{formatSleeveLabel(row.sleeve)}</td>
                      <td className="apex-kpi-value px-3 py-2 text-foreground">{row.predictions}</td>
                      <td className="apex-kpi-value px-3 py-2 text-foreground">{(row.predicted_hit_rate * 100).toFixed(1)}%</td>
                      <td className="apex-kpi-value px-3 py-2 text-foreground">{(row.realized_hit_rate * 100).toFixed(1)}%</td>
                      <td className={`apex-kpi-value px-3 py-2 ${Math.abs(row.calibration_gap) <= row.threshold_abs_gap ? "text-positive" : "text-warning"}`}>
                        {(row.calibration_gap * 100).toFixed(1)}%
                      </td>
                      <td className="apex-kpi-value px-3 py-2 text-foreground">{(row.threshold_abs_gap * 100).toFixed(1)}%</td>
                      <td className={`px-3 py-2 ${row.within_threshold ? "text-positive" : "text-negative"}`}>
                        {row.within_threshold ? "pass" : "breach"}
                      </td>
                      <td className="px-3 py-2 text-muted-foreground">{row.data_quality}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {calibration?.notes?.length ? (
            <ul className="mt-2 space-y-1 text-[11px] text-muted-foreground">
              {calibration.notes.slice(0, 2).map((note) => (
                <li key={note}>- {note}</li>
              ))}
            </ul>
          ) : null}
        </div>

        {/* ── Monthly Risk Report ── */}
        <div className="mt-4 rounded-xl border border-border/80 bg-background/60 p-3">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-sm font-semibold text-foreground">Monthly Model Risk Report</p>
            <Button
              size="sm"
              variant="outline"
              className="h-9 rounded-full px-3 text-xs"
              onClick={() => {
                void fetchMonthlyRiskReport();
              }}
              disabled={riskReportLoading}
            >
              {riskReportLoading ? "Refreshing..." : "Refresh"}
            </Button>
          </div>
          {riskReportError ? (
            <p className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {riskReportError}
            </p>
          ) : null}
          {monthlyRiskReport ? (
            <div className="space-y-2 text-xs text-muted-foreground">
              <p>
                Month: <span className="font-semibold text-foreground">{monthlyRiskReport.month}</span> | Policy changes:{" "}
                <span className="font-semibold text-foreground">{monthlyRiskReport.policy_changes.length}</span>
              </p>
              <p>
                Miss-reason buckets: <span className="font-semibold text-foreground">{Object.keys(monthlyRiskReport.miss_reasons).length}</span>
              </p>
              <p>
                Drift sleeves tracked: <span className="font-semibold text-foreground">{monthlyRiskReport.drift_rows.length}</span>
              </p>
              <div className="rounded-lg border border-border/70 bg-background/70 p-2">
                <p className="mb-1 font-semibold text-foreground">Top drift sleeves</p>
                {(monthlyRiskReport.drift_rows || []).slice(0, 3).map((row) => (
                  <p key={row.sleeve}>
                    {formatSleeveLabel(row.sleeve)}: mean gap {(row.mean_abs_gap * 100).toFixed(1)}% (thr {(row.threshold_abs_gap * 100).toFixed(1)}%)
                  </p>
                ))}
              </div>
              <div className="rounded-lg border border-border/70 bg-background/70 p-2">
                <p className="mb-1 font-semibold text-foreground">Recent policy changes</p>
                {(monthlyRiskReport.policy_changes || []).slice(0, 3).map((event) => (
                  <p key={`${event.timestamp}-${event.workflow_id}`}>
                    {event.workflow_id}: {event.from_status} {"->"} {event.to_status} ({new Date(event.timestamp).toLocaleDateString()})
                  </p>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              Refresh to load monthly drift, miss reasons, and policy-change audit events.
            </p>
          )}
        </div>
      </article>
    </section>
  );
}
