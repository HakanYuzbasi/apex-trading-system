"use client";

import { useCallback, useMemo, useState } from "react";
import { RefreshCw, ShieldCheck, Zap, Activity, History, Info, AlertCircle, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Select } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { cn, getToneClass } from "@/lib/utils";
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

function mandateBandVariant(band: string): "positive" | "warning" | "negative" {
  if (band === "green") return "positive";
  if (band === "yellow") return "warning";
  return "negative";
}

function workflowStatusVariant(status: MandateWorkflowPack["status"]): "default" | "positive" | "secondary" | "warning" {
  if (status === "paper_live") return "default";
  if (status === "approved") return "positive";
  if (status === "retired") return "secondary";
  return "warning";
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
    <section className="grid grid-cols-1 gap-6">
      <article className="glass-card apex-fade-up rounded-2xl p-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Zap className="h-5 w-5 text-primary" />
              <h2 className="text-xl font-bold text-foreground">AI Mandate Copilot</h2>
            </div>
            <p className="text-sm text-muted-foreground max-w-2xl">
              Goal-to-policy assessment with <span className="text-foreground font-medium">DD/CVaR constraints</span>. Approved plans can be applied to paper runtime controls.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="bg-primary/5 border-primary/20 text-primary h-8">
              POLICY_ONLY Mode
            </Badge>
            <Button
              size="sm"
              variant="outline"
              className="h-8 rounded-full px-4 text-xs font-semibold"
              onClick={() => {
                setHistoryDrawerOpen((prev) => !prev);
                if (!historyDrawerOpen) {
                  void fetchMandateHistory();
                }
              }}
            >
              <History className="h-3.5 w-3.5 mr-1.5" />
              {historyDrawerOpen ? "Hide History" : "Assessment History"}
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-8 rounded-full px-4 text-xs font-semibold"
              onClick={() => {
                void fetchMandateCalibration();
              }}
              disabled={calibrationLoading}
            >
              {calibrationLoading ? <RefreshCw className="h-3.5 w-3.5 animate-spin mr-1.5" /> : <Activity className="h-3.5 w-3.5 mr-1.5" />}
              {calibrationLoading ? "Syncing..." : "Sync Calibration"}
            </Button>
          </div>
        </div>

        <div className="mt-6 grid gap-4 lg:grid-cols-[1fr_220px_220px_auto] lg:items-end">
          <div className="space-y-2">
            <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <ShieldCheck className="h-3.5 w-3.5" /> Mandate Intent Prompt
            </label>
            <Textarea
              aria-label="Mandate prompt"
              className="min-h-[100px] bg-background/40 backdrop-blur-sm"
              value={mandateIntent}
              onChange={(event) => setMandateIntent(event.target.value)}
              placeholder="Example: I want to achieve 15% annualized return with high exposure to AI/SaaS sectors and tight risk limits."
            />
          </div>
          
          <div className="space-y-2">
            <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Suitability Profile</label>
            <Select
              aria-label="Suitability profile"
              className="h-10 bg-background/40 backdrop-blur-sm px-4"
              value={mandateSuitability}
              onChange={(event) => setMandateSuitability(event.target.value as SuitabilityProfile)}
            >
              <option value="conservative">Conservative (10% DD)</option>
              <option value="balanced">Balanced (15% DD)</option>
              <option value="aggressive">Aggressive (25% DD)</option>
            </Select>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Drawdown Constraint</label>
              <label className="flex items-center gap-1.5 text-[10px] font-semibold text-muted-foreground cursor-pointer">
                <input
                  aria-label="Use profile drawdown default"
                  type="checkbox"
                  className="rounded border-border bg-background"
                  checked={useProfileDrawdown}
                  onChange={(event) => setUseProfileDrawdown(event.target.checked)}
                />
                Use Profile Default
              </label>
            </div>
            <div className="relative">
              <Input
                aria-label="Max drawdown percent"
                type="number"
                min={5}
                max={50}
                step={0.5}
                disabled={useProfileDrawdown}
                className="h-10 bg-background/40 backdrop-blur-sm pr-10"
                value={mandateDrawdownCap}
                onChange={(event) => setMandateDrawdownCap(Number(event.target.value))}
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-bold text-muted-foreground">%</span>
            </div>
          </div>

          <Button
            className="h-10 px-8 font-bold rounded-xl"
            onClick={() => {
              void evaluateMandate();
            }}
            disabled={mandateLoading}
          >
            {mandateLoading ? <RefreshCw className="h-4 w-4 animate-spin mr-2" /> : < Zap className="h-4 w-4 mr-2" />}
            {mandateLoading ? "Evaluating..." : "Generate Policy"}
          </Button>
        </div>

        {mandateError && (
          <div className="mt-4 flex items-center gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive animate-in shake duration-300">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="font-semibold">{mandateError}</span>
          </div>
        )}

        {historyDrawerOpen && (
          <div className="mt-6 rounded-2xl border border-border/50 bg-background/40 backdrop-blur-md p-4 animate-in slide-in-from-top-4 duration-300">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-sm font-bold text-foreground">Assessment Audit Log</h3>
              <Badge variant="secondary" className="bg-background/50">{mandateHistory.length} events recorded</Badge>
            </div>
            
            {mandateHistoryError && (
              <div className="mb-4 rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                {mandateHistoryError}
              </div>
            )}
            
            <div className="max-h-[300px] space-y-3 overflow-y-auto pr-2 custom-scrollbar">
              {mandateHistoryLoading ? (
                <div className="flex flex-col items-center py-8 gap-2 text-muted-foreground opacity-50">
                  <RefreshCw className="h-6 w-6 animate-spin" />
                  <span className="text-xs font-medium">Synchronizing audit log...</span>
                </div>
              ) : mandateHistory.length === 0 ? (
                <div className="text-center py-10 text-xs text-muted-foreground border border-dashed rounded-xl">
                  No previous mandate assessments found.
                </div>
              ) : (
                mandateHistory.map((event) => (
                  <div key={`${event.output_hash}-${event.timestamp}`} className="glass-card hover:border-primary/30 rounded-xl px-4 py-3 transition-colors">
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                       <Badge variant={mandateBandVariant(event.response_summary.feasibility_band || "red")}>
                        {event.response_summary.feasibility_band || "unknown"}
                      </Badge>
                      <span className="text-[11px] font-mono text-muted-foreground">{new Date(event.timestamp).toLocaleString()}</span>
                    </div>
                    <p className="line-clamp-2 text-xs text-foreground font-medium mb-2">
                      {event.request?.intent || "No intent text captured."}
                    </p>
                    <div className="flex flex-wrap gap-x-4 gap-y-1">
                      <div className="flex items-center gap-1.5 min-w-[100px]">
                        <span className="text-[10px] font-bold text-muted-foreground uppercase">P(Hit)</span>
                        <span className="text-xs font-bold text-foreground">{(((event.response_summary.probability_target_hit || 0) as number) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex items-center gap-1.5 min-w-[100px]">
                        <span className="text-[10px] font-bold text-muted-foreground uppercase">Confidence</span>
                        <span className="text-xs font-bold text-foreground">{(((event.response_summary.confidence || 0) as number) * 100).toFixed(1)}%</span>
                      </div>
                       <div className="flex items-center gap-1.5 min-w-[100px]">
                        <span className="text-[10px] font-bold text-muted-foreground uppercase">Max DD</span>
                        <span className="text-xs font-bold text-negative">{Number(event.response_summary.expected_max_drawdown_pct || 0).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {mandateResult && (
          <div className="mt-8 pt-8 border-t border-border/40 animate-in fade-in duration-500">
             <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {/* Feasibility Block */}
              <div className="glass-card rounded-xl p-4 bg-background/20">
                <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-3">Feasibility Status</p>
                <div className="flex items-center gap-3">
                  <Badge variant={mandateBandVariant(mandateResult.feasibility_band)} className="h-6 px-3">
                    {mandateResult.feasibility_band}
                  </Badge>
                  <span className="text-xs font-bold text-foreground">{mandateResult.feasible ? "Aligned" : "Violated"}</span>
                </div>
              </div>

              {/* Confidence Block */}
              <div className="glass-card rounded-xl p-4 bg-background/20">
                <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-2">P(Target Hit)</p>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold tracking-tight text-foreground">{(mandateResult.probability_target_hit * 100).toFixed(1)}%</span>
                  <span className="text-[10px] font-bold text-muted-foreground">CONFIDENCE {(mandateResult.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Risk Projection Block */}
              <div className="glass-card rounded-xl p-4 bg-background/20">
                <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-2">Max Risk Projections</p>
                <div className="flex flex-col">
                  <span className={cn("text-lg font-bold", getToneClass("negative"))}>DD {mandateResult.expected_max_drawdown_pct.toFixed(1)}%</span>
                  <span className="text-[10px] font-bold text-muted-foreground">CVAR95 {mandateResult.expected_cvar95_pct.toFixed(1)}%</span>
                </div>
              </div>

              {/* Summary Block */}
              <div className="glass-card rounded-xl p-4 bg-background/20">
                <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-2">Parsed Mandate</p>
                <div className="space-y-1">
                   <div className="flex justify-between items-center text-xs">
                    <span className="text-muted-foreground">Target Ret:</span>
                    <span className="font-bold text-positive">{(Number(mandateResult.parsed_mandate?.target_return_pct ?? 0) || 0).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-muted-foreground">Horizon:</span>
                    <span className="font-bold">{Math.trunc(Number(mandateResult.parsed_mandate?.horizon_days ?? 0) || 0)} days</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              <div className="glass-card rounded-xl p-5">
                <h4 className="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-4 flex items-center gap-2">
                  <Activity className="h-4 w-4" /> Recommended Sleeve Allocations
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(mandateResult.policy.sleeve_allocations).map(([sleeve, weight]) => (
                    <div key={sleeve} className="flex items-center justify-between p-3 rounded-xl bg-background/40 border border-border/50">
                      <span className="text-xs font-bold text-foreground capitalize">{formatSleeveLabel(sleeve)}</span>
                      <Badge variant="outline" className="font-mono text-[11px] bg-primary/5 text-primary border-primary/20">
                        {(Number(weight) * 100).toFixed(1)}%
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass-card rounded-xl p-5">
                <h4 className="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-4 flex items-center gap-2">
                  <Info className="h-4 w-4" /> Strategy Rationale & Constraints
                </h4>
                <div className="space-y-4">
                  <ul className="space-y-2">
                    {mandateResult.rationale.slice(0, 3).map((item, idx) => (
                      <li key={idx} className="text-xs text-foreground flex items-start gap-2 leading-relaxed">
                        <span className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                        {item}
                      </li>
                    ))}
                  </ul>
                   <div className="pt-4 mt-4 border-t border-border/30">
                     <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-2">Policy Constraints</p>
                     <div className="flex flex-wrap gap-2">
                       {mandateResult.policy.constraints.slice(0, 4).map((c, i) => (
                         <Badge key={i} variant="secondary" className="bg-background/80 text-[10px] rounded-md px-2 py-1">
                           {c}
                         </Badge>
                       ))}
                     </div>
                   </div>
                </div>
              </div>
            </div>
            
            <div className="mt-4 bg-muted/20 rounded-xl p-3 border border-border/30 text-center">
              <p className="text-[10px] leading-relaxed italic text-muted-foreground">
                <span className="font-bold text-muted-foreground/80 tracking-widest uppercase mb-1 block">Institutional Disclaimer</span>
                {mandateResult.disclaimer}
              </p>
            </div>
          </div>
        )}

        {/* ── Workflow Pack Section ── */}
        <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_350px]">
          <div className="glass-card rounded-2xl p-6">
            <div className="flex items-center justify-between mb-6">
               <div className="flex items-center gap-2">
                 <ShieldCheck className="h-5 w-5 text-primary" />
                 <h3 className="text-base font-bold text-foreground">Advisor Workflow Control</h3>
               </div>
               <div className="flex gap-2">
                 <Button
                  size="sm"
                  variant="outline"
                  className="h-8 rounded-full text-[11px] font-bold"
                  onClick={() => void fetchWorkflowList()}
                  disabled={workflowListLoading}
                >
                   <RefreshCw className={cn("h-3 w-3 mr-1.5", workflowListLoading && "animate-spin")} />
                  Refresh Sync
                </Button>
               </div>
            </div>

            {workflowError && (
              <div className="mb-4 flex items-center gap-2 rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-xs text-destructive">
                <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                {workflowError}
              </div>
            )}

            {workflowPack ? (
              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 rounded-xl bg-background/40 border border-border/50">
                  <div className="space-y-1">
                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">Active Workflow ID</p>
                    <p className="text-sm font-mono font-bold text-foreground">{workflowPack.workflow_id}</p>
                  </div>
                  <Badge variant={workflowStatusVariant(workflowPack.status)} className="h-7 px-4">
                    {workflowPack.status}
                  </Badge>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className={cn("p-4 rounded-xl border transition-colors", pmSigned ? "bg-positive/5 border-positive/30" : "bg-warning/5 border-warning/30")}>
                    <p className="text-[10px] font-bold text-muted-foreground uppercase mb-2">PM Sign-off</p>
                    <div className="flex items-center justify-between">
                       <span className={cn("text-xs font-bold", pmSigned ? "text-positive" : "text-warning")}>
                         {pmSigned ? "VERIFIED" : "PENDING"}
                       </span>
                       <Button
                        size="sm"
                        className="h-7 px-3 text-[10px] font-bold"
                        onClick={() => void signoffWorkflow(workflowPack.workflow_id, "pm")}
                        disabled={workflowLoading || pmSigned}
                      >
                        {pmSigned ? "Signed" : "Sign as PM"}
                      </Button>
                    </div>
                  </div>
                  <div className={cn("p-4 rounded-xl border transition-colors", complianceSigned ? "bg-positive/5 border-positive/30" : "bg-warning/5 border-warning/30")}>
                    <p className="text-[10px] font-bold text-muted-foreground uppercase mb-2">Compliance Review</p>
                    <div className="flex items-center justify-between">
                       <span className={cn("text-xs font-bold", complianceSigned ? "text-positive" : "text-warning")}>
                         {complianceSigned ? "VERIFIED" : "PENDING"}
                       </span>
                       <Button
                        size="sm"
                        className="h-7 px-3 text-[10px] font-bold"
                        onClick={() => void signoffWorkflow(workflowPack.workflow_id, "compliance")}
                        disabled={workflowLoading || complianceSigned}
                      >
                        {complianceSigned ? "Signed" : "Sign as L&C"}
                      </Button>
                    </div>
                  </div>
                </div>

                <div className="flex gap-3">
                   <Button
                    className="flex-1 h-10 font-bold"
                    onClick={() => void updateWorkflowStatus(workflowPack.workflow_id, "paper_live", "Applied to paper runtime.")}
                    disabled={workflowLoading || workflowPack.status !== "approved"}
                  >
                    Deploy to Paper Runtime
                  </Button>
                   <Button
                    variant="outline"
                    className="h-10 px-6 font-bold text-destructive hover:bg-destructive/5"
                    onClick={() => void updateWorkflowStatus(workflowPack.workflow_id, "retired", "Retired from review.")}
                    disabled={workflowLoading || workflowPack.status === "retired"}
                  >
                    Retire Pack
                  </Button>
                </div>
              </div>
            ) : (
              <div className="py-12 flex flex-col items-center gap-4 text-center border-2 border-dashed border-border/40 rounded-2xl">
                <div className="h-12 w-12 rounded-full bg-muted/40 flex items-center justify-center">
                   <Lock className="h-6 w-6 text-muted-foreground" />
                </div>
                <div className="max-w-xs space-y-2">
                  <p className="text-sm font-bold text-foreground">No Active Workflow</p>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Generate a mandate policy first to initiate an institutional approval workflow.
                  </p>
                </div>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    className="h-8 rounded-full px-6 font-bold"
                    onClick={() => void initiateWorkflowPack()}
                    disabled={workflowLoading || !mandateResult?.request_id}
                  >
                    Initiate Workflow
                  </Button>
                   <Button
                    size="sm"
                    variant="secondary"
                    className="h-8 rounded-full px-6 font-bold"
                    onClick={() => void activatePlanFlow()}
                    disabled={workflowLoading || !mandateResult?.request_id}
                  >
                    Auto-Activate
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div className="glass-card rounded-2xl border border-border/50 overflow-hidden flex flex-col">
             <div className="px-4 py-3 border-b border-border/40 bg-background/40 flex items-center justify-between">
                <h4 className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Workflow Audit History</h4>
                <Badge variant="secondary" className="text-[10px] bg-background/50">{workflowList.length} packs</Badge>
             </div>
             <div className="flex-1 overflow-y-auto max-h-[350px] custom-scrollbar">
                {workflowList.length === 0 ? (
                  <div className="p-8 text-center text-[11px] text-muted-foreground opacity-50 italic">
                    No historical workflow data.
                  </div>
                ) : (
                  workflowList.map((row) => (
                    <button
                      key={row.workflow_id}
                      className={cn(
                        "w-full px-4 py-3 flex flex-col gap-1 text-left border-b border-border/30 hover:bg-primary/5 transition-colors",
                        workflowPack?.workflow_id === row.workflow_id && "bg-primary/[0.03] border-l-2 border-l-primary"
                      )}
                      onClick={() => setWorkflowPack(row)}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-mono font-bold text-foreground">{row.workflow_id.slice(0, 12)}...</span>
                        <Badge variant={workflowStatusVariant(row.status)} className="text-[9px] h-4 px-1.5 uppercase">
                          {row.status}
                        </Badge>
                      </div>
                      <span className="text-[10px] text-muted-foreground">{new Date(row.created_at).toLocaleDateString()} • {row.created_by}</span>
                    </button>
                  ))
                )}
             </div>
          </div>
        </div>

        {/* ── Calibration and Risk Logs ── */}
        <div className="mt-8 grid gap-6 lg:grid-cols-2">
           {/* Calibration Table */}
           <div className="glass-card rounded-2xl p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                   <Activity className="h-5 w-5 text-primary" />
                   <h3 className="text-base font-bold text-foreground">Sleeve Calibration Analytics</h3>
                </div>
                <span className="text-[10px] font-bold text-muted-foreground uppercase bg-muted/30 px-2 py-1 rounded">
                  {calibration?.lookback_events ?? 0} events
                </span>
              </div>
              
              <div className="overflow-hidden border border-border/50 rounded-xl bg-background/20 backdrop-blur-sm">
                <table className="w-full text-[11px]">
                   <thead className="bg-background/60 text-muted-foreground">
                      <tr>
                        <th className="px-3 py-3 font-bold text-left uppercase tracking-tighter">Sleeve</th>
                        <th className="px-3 py-3 font-bold text-center uppercase tracking-tighter">N</th>
                        <th className="px-3 py-3 font-bold text-center uppercase tracking-tighter">Predicted</th>
                        <th className="px-3 py-3 font-bold text-center uppercase tracking-tighter">Realized</th>
                        <th className="px-3 py-3 font-bold text-center uppercase tracking-tighter">Gap</th>
                        <th className="px-3 py-3 font-bold text-center uppercase tracking-tighter">Status</th>
                      </tr>
                   </thead>
                   <tbody className="divide-y divide-border/30">
                     {calibrationLoading ? (
                       <tr><td colSpan={6} className="py-12 text-center text-muted-foreground animate-pulse">Computing calibration vectors...</td></tr>
                     ) : (calibration?.rows ?? []).map((row) => (
                       <tr key={row.sleeve} className="hover:bg-background/30 transition-colors">
                         <td className="px-3 py-3 font-bold text-foreground capitalize">{formatSleeveLabel(row.sleeve)}</td>
                         <td className="px-3 py-3 text-center">{row.predictions}</td>
                         <td className="px-3 py-3 text-center">{(row.predicted_hit_rate * 100).toFixed(1)}%</td>
                         <td className="px-3 py-3 text-center font-bold">{(row.realized_hit_rate * 100).toFixed(1)}%</td>
                         <td className={cn("px-3 py-3 text-center font-mono", Math.abs(row.calibration_gap) <= row.threshold_abs_gap ? "text-positive" : "text-warning")}>
                           {(row.calibration_gap * 100).toFixed(1)}%
                         </td>
                         <td className="px-3 py-3 text-center">
                            <Badge variant={row.within_threshold ? "positive" : "negative"} className="text-[9px] h-4 px-1 rounded">
                              {row.within_threshold ? "PASS" : "BREACH"}
                            </Badge>
                         </td>
                       </tr>
                     ))}
                   </tbody>
                </table>
              </div>
           </div>

           {/* Monthly Risk Report */}
           <div className="glass-card rounded-2xl p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                   <Info className="h-5 w-5 text-primary" />
                   <h3 className="text-base font-bold text-foreground">Monthly Risk Audit</h3>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 rounded-full text-[11px] font-bold"
                  onClick={() => void fetchMonthlyRiskReport()}
                  disabled={riskReportLoading}
                >
                  <RefreshCw className={cn("h-3 w-3 mr-1.5", riskReportLoading && "animate-spin")} />
                  Audit Refresh
                </Button>
              </div>

              {monthlyRiskReport ? (
                <div className="space-y-4">
                   <div className="grid grid-cols-3 gap-2">
                      <div className="bg-background/40 p-3 rounded-xl border border-border/40 text-center">
                         <p className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Audit Month</p>
                         <p className="text-sm font-bold text-foreground">{monthlyRiskReport.month}</p>
                      </div>
                      <div className="bg-background/40 p-3 rounded-xl border border-border/40 text-center">
                         <p className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Policy Deltas</p>
                         <p className="text-sm font-bold text-foreground">{monthlyRiskReport.policy_changes.length}</p>
                      </div>
                      <div className="bg-background/40 p-3 rounded-xl border border-border/40 text-center">
                         <p className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Sleeves Tracked</p>
                         <p className="text-sm font-bold text-foreground">{monthlyRiskReport.drift_rows.length}</p>
                      </div>
                   </div>

                   <div className="p-4 rounded-xl bg-background/20 border border-border/30">
                      <p className="text-[10px] font-bold text-muted-foreground uppercase mb-3 tracking-widest">High-Drift Vectors</p>
                      <div className="space-y-2">
                         {monthlyRiskReport.drift_rows.slice(0, 3).map(row => (
                           <div key={row.sleeve} className="flex justify-between items-center text-xs">
                             <span className="font-medium text-foreground capitalize">{formatSleeveLabel(row.sleeve)}</span>
                             <span className="font-mono text-[11px]">GAP: {(row.mean_abs_gap * 100).toFixed(1)}% <span className="text-muted-foreground">/ THR {(row.threshold_abs_gap * 100).toFixed(1)}%</span></span>
                           </div>
                         ))}
                      </div>
                   </div>

                   <div className="p-4 rounded-xl bg-background/20 border border-border/30">
                      <p className="text-[10px] font-bold text-muted-foreground uppercase mb-3 tracking-widest">Policy State Transitions</p>
                      <div className="space-y-2">
                         {monthlyRiskReport.policy_changes.slice(0, 3).map(event => (
                           <div key={`${event.timestamp}-${event.workflow_id}`} className="flex justify-between items-center text-[11px]">
                             <span className="font-bold text-foreground">{event.workflow_id.slice(0, 8)}</span>
                             <span className="flex items-center gap-1.5 font-bold">
                               <Badge variant="secondary" className="text-[9px] h-4">{event.from_status}</Badge>
                               <span className="opacity-50">→</span>
                               <Badge variant="positive" className="text-[9px] h-4">{event.to_status}</Badge>
                             </span>
                           </div>
                         ))}
                      </div>
                   </div>
                </div>
              ) : (
                <div className="py-20 flex flex-col items-center gap-2 text-center opacity-50 grayscale">
                   <Info className="h-8 w-8 text-muted-foreground" />
                   <p className="text-[11px] font-medium max-w-[200px]">Historical drift audit logs require synchronizing with the model risk engine.</p>
                </div>
              )}
           </div>
        </div>
      </article>
    </section>
  );
}
