"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  AlertTriangle,
  BarChart3,
  DollarSign,
  Gauge,
  LogOut,
  Moon,
  RefreshCw,
  ShieldCheck,
  Sun,
  Timer,
  TrendingDown,
  TrendingUp,
  Wifi,
  WifiOff,
} from "lucide-react";
import {
  useCockpitData,
  useMetrics,
  type CockpitAlert,
  type CockpitDerivative,
  type CockpitPosition,
  type SocialAuditEvent,
  type SleeveAttribution,
} from "@/lib/api";
import { useTheme } from "@/components/theme/ThemeProvider";
import { useAuthContext } from "@/components/auth/AuthProvider";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import AlertsFeed from "@/components/dashboard/AlertsFeed";
import ControlsPanel from "@/components/dashboard/ControlsPanel";
import EquityPanel from "@/components/dashboard/EquityPanel";
import PositionsTable from "@/components/dashboard/PositionsTable";
import ExplainableAIChart from "@/components/dashboard/ExplainableAIChart";
import { Skeleton } from "@/components/ui/skeleton";
import { useWebSocket } from "@/hooks/useWebSocket";
import {
  sanitizeCount,
  sanitizeExecutionMetrics,
  sanitizeMoney,
} from "@/lib/metricGuards";
import {
  MAX_POSITIONS,
  DRAWDOWN_BUDGET_PCT,
  SHARPE_TARGET,
  WIN_RATE_TARGET,
  RETURN_CYCLE_TARGET,
  TRADE_CYCLE_TARGET,
  EDGE_CAPTURE_TARGET,
} from "@/lib/constants";
import {
  clampPct,
  formatCurrency,
  formatCurrencyWithCents,
  formatCompactCurrency,
  formatPct,
  normalizeDrawdownPct,
  formatSleeveLabel,
  sortIndicator,
} from "@/lib/formatters";

type LensKey = "performance" | "risk" | "execution";
type SortDirection = "asc" | "desc";
type PositionSortKey = "symbol" | "qty" | "entry" | "current" | "pnl" | "pnl_pct" | "signal_direction";
type SeverityFilter = "all" | "critical" | "warning" | "info";

type LensRow = {
  label: string;
  value: string;
  hint: string;
  tone?: "neutral" | "positive" | "negative";
};

type LensBar = {
  label: string;
  value: number;
  targetLabel: string;
  tone?: "neutral" | "positive" | "negative";
};

type LensModel = {
  title: string;
  subtitle: string;
  rows: LensRow[];
  bars: LensBar[];
};

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


function toneClass(tone: LensRow["tone"]): string {
  if (tone === "positive") return "text-positive";
  if (tone === "negative") return "text-negative";
  return "text-foreground";
}

function barToneClass(tone: LensBar["tone"]): string {
  if (tone === "positive") return "bg-positive";
  if (tone === "negative") return "bg-negative";
  return "bg-primary";
}

function severityBadgeClass(severity: CockpitAlert["severity"]): string {
  if (severity === "critical") return "bg-negative/15 text-negative";
  if (severity === "warning") return "bg-warning/15 text-warning";
  return "bg-primary/15 text-primary";
}

function comparePositions(a: CockpitPosition, b: CockpitPosition, key: PositionSortKey): number {
  if (key === "symbol" || key === "signal_direction") {
    return String(a[key]).localeCompare(String(b[key]));
  }
  return Number(a[key]) - Number(b[key]);
}

function positionRowKey(position: CockpitPosition): string {
  const symbol = String(position.symbol ?? "").trim().toUpperCase();
  const source = String(position.source_id ?? "").trim()
    || String(position.broker_type ?? "").trim().toLowerCase()
    || "state";
  const sideRaw = String(position.side ?? (position.qty < 0 ? "SHORT" : "LONG")).trim().toUpperCase();
  const side = sideRaw === "SHORT" ? "SHORT" : "LONG";
  const qty = Number.isFinite(Number(position.qty)) ? Math.trunc(Number(position.qty)) : 0;
  const securityType = String((position as Record<string, unknown>).security_type ?? "").trim().toUpperCase();
  const expiry = String((position as Record<string, unknown>).expiry ?? "").trim().toUpperCase();
  const strike = Number.isFinite(Number((position as Record<string, unknown>).strike))
    ? Number((position as Record<string, unknown>).strike).toFixed(4)
    : "0.0000";
  const right = String((position as Record<string, unknown>).right ?? "").trim().toUpperCase();
  return `${symbol}|${source}|${side}|${securityType}|${expiry}|${strike}|${right}|${qty}`;
}


function mandateBandClass(band: string): string {
  if (band === "green") return "bg-positive/15 text-positive";
  if (band === "yellow") return "bg-warning/15 text-warning";
  return "bg-negative/15 text-negative";
}

function readinessClass(state: "ok" | "warn" | "down"): string {
  if (state === "ok") return "bg-positive/15 text-positive";
  if (state === "warn") return "bg-warning/15 text-warning";
  return "bg-negative/15 text-negative";
}

function workflowStatusClass(status: MandateWorkflowPack["status"]): string {
  if (status === "paper_live") return "bg-primary/15 text-primary";
  if (status === "approved") return "bg-positive/15 text-positive";
  if (status === "retired") return "bg-muted text-muted-foreground";
  return "bg-warning/15 text-warning";
}

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

export default function Dashboard({ isPublic = false }: { isPublic?: boolean }) {
  const router = useRouter();
  const { logout } = useAuthContext();
  const { isConnected: wsConnected, lastMessage: wsMessage } = useWebSocket(isPublic);
  const { metrics, isLoading: metricsLoading, isError: metricsError, error: metricsFetchError } = useMetrics(isPublic);
  const { data: cockpit, isLoading: cockpitLoading, isError: cockpitError, error: cockpitFetchError } = useCockpitData(isPublic);
  const { theme, toggleTheme } = useTheme();

  // Alert management state
  const [dismissedAlerts, setDismissedAlerts] = useState<Record<string, boolean>>({});
  const [alertFilter, setAlertFilter] = useState<SeverityFilter>("all");

  // UI state
  const [themeMounted, setThemeMounted] = useState(false);
  const [activeLens, setActiveLens] = useState<LensKey>("performance");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [historyDrawerOpen, setHistoryDrawerOpen] = useState(false);

  // Multi-broker aggregated equity
  const [aggregatedEquity, setAggregatedEquity] = useState<number | null>(null);
  const [brokerCount, setBrokerCount] = useState<number>(0);

  // Account selector — broker source list and selected account
  const [brokerSources, setBrokerSources] = useState<{ id: string; name: string; broker_type: string; environment: string }[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);

  // Position table sorting state
  const [sortKey, setSortKey] = useState<PositionSortKey>("symbol");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");
  const [positionsSnapshot, setPositionsSnapshot] = useState<CockpitPosition[]>([]);
  const [derivativesSnapshot, setDerivativesSnapshot] = useState<CockpitDerivative[]>([]);
  const [sleevesSnapshot, setSleevesSnapshot] = useState<SleeveAttribution[]>([]);
  const [socialAuditSnapshot, setSocialAuditSnapshot] = useState<SocialAuditEvent[]>([]);

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
  const [sessionExpiredRedirecting, setSessionExpiredRedirecting] = useState(false);

  // ... existing state ...

  const handleSessionExpired = useCallback(() => {
    if (isPublic || sessionExpiredRedirecting) {
      return;
    }
    setSessionExpiredRedirecting(true);
    logout();
    router.replace("/login?reason=session_expired");
  }, [isPublic, logout, router, sessionExpiredRedirecting]);

  // Parse WebSocket message
  const wsData = wsMessage?.type === "state_update" ? wsMessage : null;

  // Extract SHAP values from WS or fall back to mock data for demonstration
  const activeShapData = (wsData as any)?.shap_values ?? {
    "Price Momentum (Feature #2)": 0.45,
    "Volatility Profile (Feature #1)": -0.12,
    "RSI State (Feature #4)": 0.08,
    "MACD Action (Feature #0)": -0.05,
    "Statistical Z-Score (Feature #3)": 0.02
  };

  // Merge Metrics: Prefer WS data, fallback to REST
  const mergedMetrics = useMemo(() => {
    const cockpitStatus = cockpit?.status;
    const baseline = cockpitStatus
      ? {
        status: cockpitStatus.state_fresh,
        timestamp: cockpitStatus.timestamp,
        capital: cockpitStatus.capital,
        starting_capital: cockpitStatus.starting_capital ?? 0,
        daily_pnl: cockpitStatus.daily_pnl,
        total_pnl: cockpitStatus.total_pnl,
        max_drawdown: cockpitStatus.max_drawdown,
        sharpe_ratio: cockpitStatus.sharpe_ratio,
        win_rate: cockpitStatus.win_rate,
        open_positions: cockpitStatus.open_positions,
        option_positions: cockpitStatus.option_positions,
        open_positions_total: cockpitStatus.open_positions_total,
        trades_count: cockpitStatus.total_trades,
      }
      : metrics;
    if (!wsData) {
      return baseline;
    }

    const baselineRecord = (baseline ?? {}) as Record<string, unknown>;
    const wsCapital = sanitizeMoney(
      wsData.aggregated_equity ?? wsData.total_equity ?? wsData.capital,
      Number.NaN,
    );
    const mergeZeroGuard = (primary: unknown, fallback: unknown): number => {
      const primaryNum = Number(primary);
      const fallbackNum = Number(fallback);
      const safePrimary = Number.isFinite(primaryNum) ? primaryNum : 0;
      const safeFallback = Number.isFinite(fallbackNum) ? fallbackNum : 0;
      if (Math.abs(safePrimary) <= 1e-9 && Math.abs(safeFallback) > 1e-9) {
        return safeFallback;
      }
      return safePrimary;
    };

    return {
      status: true,
      timestamp: wsData.timestamp ?? baselineRecord.timestamp ?? null,
      capital: Number.isFinite(wsCapital) ? wsCapital : sanitizeMoney(baselineRecord.capital, 0),
      starting_capital: sanitizeMoney(
        wsData.starting_capital ?? wsData.initial_capital,
        sanitizeMoney(baselineRecord.starting_capital, 0),
      ),
      daily_pnl: mergeZeroGuard(wsData.daily_pnl, baselineRecord.daily_pnl),
      total_pnl: mergeZeroGuard(wsData.total_pnl, baselineRecord.total_pnl),
      max_drawdown: mergeZeroGuard(wsData.max_drawdown, baselineRecord.max_drawdown),
      sharpe_ratio: mergeZeroGuard(wsData.sharpe_ratio, baselineRecord.sharpe_ratio),
      win_rate: mergeZeroGuard(wsData.win_rate, baselineRecord.win_rate),
      open_positions: Math.max(
        sanitizeCount(wsData.open_positions, 0),
        sanitizeCount(baselineRecord.open_positions, 0),
      ),
      option_positions: sanitizeCount(baselineRecord.option_positions, 0),
      open_positions_total: Math.max(
        sanitizeCount(wsData.open_positions, 0),
        sanitizeCount(baselineRecord.open_positions_total, 0),
      ),
      trades_count: Math.max(
        sanitizeCount(wsData.total_trades, 0),
        sanitizeCount(baselineRecord.trades_count, 0),
      ),
    };
  }, [wsData, metrics, cockpit?.status]);

  // Fetch aggregated equity from multi-broker balance endpoint
  useEffect(() => {
    const fetchAggregated = async () => {
      try {
        const res = await fetch("/api/v1/portfolio/balance", { cache: "no-store" });
        if (res.status === 401) {
          handleSessionExpired();
          return;
        }
        if (res.ok) {
          const data = await res.json() as { total_equity: number; breakdown?: { source: string }[] };
          setAggregatedEquity(sanitizeMoney(data.total_equity, 0));
          setBrokerCount(Math.max(1, sanitizeCount(data.breakdown?.length ?? 1, 1)));
        }
      } catch {
        // silently skip — not critical
      }
    };
    void fetchAggregated();
    const interval = setInterval(() => void fetchAggregated(), 30000);
    return () => clearInterval(interval);
  }, [handleSessionExpired]);

  // Fetch broker sources for account selector
  useEffect(() => {
    const fetchSources = async () => {
      try {
        const res = await fetch("/api/v1/portfolio/sources", { cache: "no-store" });
        if (res.status === 401) {
          handleSessionExpired();
          return;
        }
        if (res.ok) {
          const data = await res.json() as { id: string; name: string; broker_type: string; environment: string }[];
          setBrokerSources(data);
        }
      } catch {
        // silently skip
      }
    };
    void fetchSources();
  }, [handleSessionExpired]);

  const errorText = `${String((cockpitFetchError as Error | undefined)?.message ?? "")} ${String((metricsFetchError as Error | undefined)?.message ?? "")}`.toLowerCase();
  const authExpired = sessionExpiredRedirecting
    || errorText.includes("not authenticated")
    || errorText.includes("session expired")
    || errorText.includes("401");

  // ... effects ...
  useEffect(() => {
    if (authExpired) {
      handleSessionExpired();
    }
  }, [authExpired, handleSessionExpired]);

  useEffect(() => {
    setThemeMounted(true);
  }, []);

  const showLoading = (metricsLoading || cockpitLoading) && !mergedMetrics;
  const apiReachable = wsConnected || (cockpit?.status.api_reachable ?? (!cockpitError || !metricsError));
  const stateFresh = wsConnected || (cockpit?.status.state_fresh ?? Boolean(mergedMetrics?.status));
  const isDisconnected = !showLoading && !apiReachable;
  const isStale = apiReachable && !stateFresh;
  const mergedMetricRecord = (mergedMetrics ?? {}) as Record<string, unknown>;
  const mergedOpenPositions = Math.max(
    sanitizeCount(mergedMetricRecord.open_positions, 0),
    sanitizeCount(cockpit?.status.open_positions, 0),
    sanitizeCount(cockpit?.positions?.length, 0),
  );
  const mergedOptionPositions = Math.max(
    sanitizeCount(mergedMetricRecord.option_positions, 0),
    sanitizeCount(cockpit?.status.option_positions, 0),
  );
  const mergedOpenPositionsTotal = Math.max(
    sanitizeCount(mergedMetricRecord.open_positions_total, 0),
    sanitizeCount(cockpit?.status.open_positions_total, 0),
    mergedOpenPositions + mergedOptionPositions,
  );
  const sanitizedMetrics = useMemo(
    () =>
      sanitizeExecutionMetrics({
        ...(mergedMetrics ?? {}),
        open_positions: mergedOpenPositions,
        option_positions: mergedOptionPositions,
        open_positions_total: mergedOpenPositionsTotal,
      }),
    [mergedMetrics, mergedOpenPositions, mergedOptionPositions, mergedOpenPositionsTotal],
  );
  const wsAggregatedEquity = sanitizeMoney(wsData?.aggregated_equity ?? wsData?.total_equity, Number.NaN);
  const capital = Number.isFinite(wsAggregatedEquity) ? wsAggregatedEquity : sanitizedMetrics.capital;
  const dailyPnl = sanitizedMetrics.daily_pnl;
  const totalPnl = sanitizedMetrics.total_pnl;
  const sharpe = sanitizedMetrics.sharpe_ratio;
  const winRate = sanitizedMetrics.win_rate;
  const openPositions = sanitizedMetrics.open_positions;
  const tradesCount = sanitizedMetrics.trades_count;
  const drawdownPct = normalizeDrawdownPct(sanitizedMetrics.max_drawdown);


  // Parse WS positions to match CockpitPosition interface
  const cockpitPositions = useMemo(() => cockpit?.positions ?? [], [cockpit?.positions]);
  const cockpitPositionSourceHints = useMemo(() => {
    const hints = new Map<string, { source_id: string; broker_type: string }>();
    for (const position of cockpitPositions) {
      const symbol = String(position.symbol ?? "").trim().toUpperCase();
      if (!symbol) continue;
      const sideRaw = String(position.side ?? (position.qty < 0 ? "SHORT" : "LONG")).trim().toUpperCase();
      const side = sideRaw === "SHORT" ? "SHORT" : "LONG";
      const sourceId = String(position.source_id ?? "").trim();
      const brokerType = String(position.broker_type ?? "").trim().toLowerCase();
      if (!hints.has(`${symbol}|${side}`)) {
        hints.set(`${symbol}|${side}`, { source_id: sourceId, broker_type: brokerType });
      }
      if (!hints.has(symbol)) {
        hints.set(symbol, { source_id: sourceId, broker_type: brokerType });
      }
    }
    return hints;
  }, [cockpitPositions]);

  const wsPositions = useMemo(() => {
    if (!wsData?.positions) return null;
    return Object.entries(wsData.positions).map(([symbol, raw]) => {
      const data = (raw && typeof raw === "object") ? (raw as Record<string, unknown>) : {};
      const qty = Math.trunc(sanitizeMoney(data.qty, 0));
      if (qty === 0) return null;
      const symbolUpper = String(symbol).trim().toUpperCase();
      const side = qty < 0 ? "SHORT" : "LONG";
      const sourceHint = cockpitPositionSourceHints.get(`${symbolUpper}|${side}`)
        ?? cockpitPositionSourceHints.get(symbolUpper);
      const sourceId = String(data.source_id ?? sourceHint?.source_id ?? "");
      const brokerType = String(data.broker_type ?? sourceHint?.broker_type ?? "");
      return {
        symbol: symbolUpper,
        qty,
        side,
        entry: sanitizeMoney(data.avg_price, 0),
        current: sanitizeMoney(data.current_price, 0),
        pnl: sanitizeMoney(data.pnl, 0),
        pnl_pct: sanitizeMoney(data.pnl_pct, 0),
        signal: sanitizeMoney(data.current_signal, 0),
        signal_direction: String(data.signal_direction ?? "UNKNOWN"),
        broker_type: brokerType,
        stale: Boolean(data.stale),
        source_status: String(data.source_status ?? ""),
        source_id: sourceId,
      };
    }).filter((row): row is CockpitPosition => row !== null);
  }, [wsData, cockpitPositionSourceHints]);

  const positionsCandidate = useMemo(() => {
    if (wsPositions && wsPositions.length > 0) {
      return wsPositions;
    }
    if (cockpitPositions.length > 0) {
      return cockpitPositions;
    }
    return wsPositions ?? cockpitPositions;
  }, [wsPositions, cockpitPositions]);
  const positionsFeedDegraded = !apiReachable || Boolean(cockpitError) || Boolean(metricsError) || isStale;
  const confirmedFlatPositions = !positionsFeedDegraded
    && mergedOpenPositions === 0
    && (wsPositions?.length ?? 0) === 0
    && cockpitPositions.length === 0;
  useEffect(() => {
    if (positionsCandidate.length > 0) {
      setPositionsSnapshot(positionsCandidate);
      return;
    }
    if (confirmedFlatPositions) {
      setPositionsSnapshot([]);
    }
  }, [positionsCandidate, confirmedFlatPositions]);
  const positions = useMemo(() => {
    if (positionsCandidate.length > 0) {
      return positionsCandidate;
    }
    if (confirmedFlatPositions) {
      return [];
    }
    return positionsSnapshot;
  }, [positionsCandidate, confirmedFlatPositions, positionsSnapshot]);

  const cockpitDerivatives = useMemo(() => cockpit?.derivatives ?? [], [cockpit?.derivatives]);
  const optionPositionsFromStatus = sanitizeCount(cockpit?.status.option_positions, sanitizedMetrics.option_positions);
  const confirmedFlatDerivatives = !positionsFeedDegraded
    && optionPositionsFromStatus === 0
    && cockpitDerivatives.length === 0;
  useEffect(() => {
    if (cockpitDerivatives.length > 0) {
      setDerivativesSnapshot(cockpitDerivatives);
      return;
    }
    if (confirmedFlatDerivatives) {
      setDerivativesSnapshot([]);
    }
  }, [cockpitDerivatives, confirmedFlatDerivatives]);
  const derivatives = useMemo(() => {
    if (cockpitDerivatives.length > 0) {
      return cockpitDerivatives;
    }
    if (confirmedFlatDerivatives) {
      return [];
    }
    return derivativesSnapshot;
  }, [cockpitDerivatives, confirmedFlatDerivatives, derivativesSnapshot]);

  const cockpitSleeves = useMemo(() => cockpit?.attribution?.sleeves ?? [], [cockpit?.attribution?.sleeves]);
  const confirmedFlatSleeves = !positionsFeedDegraded
    && openPositions === 0
    && cockpitSleeves.length === 0;
  useEffect(() => {
    if (cockpitSleeves.length > 0) {
      setSleevesSnapshot(cockpitSleeves);
      return;
    }
    if (confirmedFlatSleeves) {
      setSleevesSnapshot([]);
    }
  }, [cockpitSleeves, confirmedFlatSleeves]);
  const sleeves = useMemo(() => {
    if (cockpitSleeves.length > 0) {
      return cockpitSleeves;
    }
    if (confirmedFlatSleeves) {
      return [];
    }
    return sleevesSnapshot;
  }, [cockpitSleeves, confirmedFlatSleeves, sleevesSnapshot]);

  const socialAudit = cockpit?.social_audit;
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
  const notes = useMemo(() => cockpit?.notes ?? [], [cockpit?.notes]);
  const usp = cockpit?.usp;
  const brokerRuntime = useMemo(() => {
    const rows = cockpit?.status?.brokers ?? [];
    const byType = new Map(rows.map((row) => [row.broker, row]));
    return {
      alpaca: byType.get("alpaca"),
      ibkr: byType.get("ibkr"),
      active: cockpit?.status?.active_broker ?? "none",
    };
  }, [cockpit?.status?.active_broker, cockpit?.status?.brokers]);

  const brokerStateToBadge = useCallback((
    state: "live" | "stale" | "configured" | "not_configured" | undefined,
    mode?: "trading" | "idle" | "disabled",
  ) => {
    if (mode === "trading" && state && state !== "not_configured") return "ok";
    if (mode === "idle") return "warn";
    if (state === "live") return "ok";
    if (state === "stale" || state === "configured") return "warn";
    return "down";
  }, []);

  const brokerDetail = useCallback((broker: {
    status: "live" | "stale" | "configured" | "not_configured";
    mode?: "trading" | "idle" | "disabled";
    source_count: number;
    live_source_count: number;
    heartbeat_ts?: string | null;
    stale_age_seconds?: number | null;
  } | undefined) => {
    if (!broker) return "not configured";
    const heartbeatAge = Number.isFinite(Number(broker.stale_age_seconds))
      ? `${Math.max(0, Math.trunc(Number(broker.stale_age_seconds)))}s`
      : "n/a";
    const heartbeatTime = broker.heartbeat_ts
      ? String(broker.heartbeat_ts).split("T")[1]?.replace(/\.\d+Z$/, "Z") ?? String(broker.heartbeat_ts)
      : "unknown";
    const heartbeatDetail = `${heartbeatAge} @ ${heartbeatTime}`;
    if (broker.mode === "idle") return `idle (positions/metrics) • hb ${heartbeatDetail}`;
    if (broker.mode === "trading") return `trading active • hb ${heartbeatDetail}`;
    if (broker.status === "live") return `${broker.live_source_count}/${Math.max(1, broker.source_count)} live • hb ${heartbeatDetail}`;
    if (broker.status === "stale") return `configured (stale ${heartbeatDetail})`;
    if (broker.status === "configured") return `configured (idle) • hb ${heartbeatDetail}`;
    return "not configured";
  }, []);
  const optionPositions = optionPositionsFromStatus;
  const totalLines = Math.max(
    sanitizeCount(cockpit?.status.open_positions_total, sanitizedMetrics.open_positions_total),
    openPositions + optionPositions,
  );

  const startingCapital = Math.max(1, sanitizedMetrics.starting_capital || (capital - totalPnl));
  const returnPct = startingCapital > 0 ? totalPnl / startingCapital : 0;
  const pnlPerTrade = tradesCount > 0 ? totalPnl / tradesCount : 0;
  const avgPositionSize = openPositions > 0 ? capital / openPositions : 0;
  const nowLabel = String(mergedMetrics?.timestamp ?? "--");

  const activeAlerts = useMemo(() => {
    const source = cockpit?.alerts ?? [];
    return source.filter((alert) => !dismissedAlerts[alert.id]);
  }, [cockpit?.alerts, dismissedAlerts]);

  const triageAlerts = useMemo(() => {
    if (alertFilter === "all") return activeAlerts;
    return activeAlerts.filter((alert) => alert.severity === alertFilter);
  }, [activeAlerts, alertFilter]);

  const sortedPositions = useMemo(() => {
    let copy = [...positions];
    if (selectedSourceId) {
      copy = copy.filter(p => p.source_id === selectedSourceId);
    }
    copy.sort((a, b) => {
      const cmp = comparePositions(a, b, sortKey);
      return sortDirection === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [positions, sortDirection, sortKey, selectedSourceId]);

  const lensModel = useMemo<Record<LensKey, LensModel>>(() => {
    return {
      performance: {
        title: "Performance Lens",
        subtitle: "Alpha quality, capital growth, and efficiency diagnostics.",
        rows: [
          {
            label: "Capital",
            value: formatCurrencyWithCents(capital),
            hint: "Live account equity",
          },
          {
            label: "Daily PnL",
            value: formatCurrency(dailyPnl),
            hint: "Intraday marked performance",
            tone: dailyPnl >= 0 ? "positive" : "negative",
          },
          {
            label: "Net PnL",
            value: formatCurrency(totalPnl),
            hint: "Strategy total PnL",
            tone: totalPnl >= 0 ? "positive" : "negative",
          },
          {
            label: "Return",
            value: formatPct(returnPct),
            hint: "PnL over starting equity",
            tone: returnPct >= 0 ? "positive" : "negative",
          },
        ],
        bars: [
          {
            label: "Sharpe Attainment",
            value: clampPct((sharpe / SHARPE_TARGET) * 100),
            targetLabel: `Target ${SHARPE_TARGET.toFixed(2)}`,
            tone: sharpe >= SHARPE_TARGET ? "positive" : "neutral",
          },
          {
            label: "Win Rate Quality",
            value: clampPct((winRate / WIN_RATE_TARGET) * 100),
            targetLabel: `Target ${(WIN_RATE_TARGET * 100).toFixed(0)}%`,
            tone: winRate >= WIN_RATE_TARGET ? "positive" : "neutral",
          },
          {
            label: "Capital Efficiency",
            value: clampPct((Math.abs(returnPct) / RETURN_CYCLE_TARGET) * 100),
            targetLabel: `${(RETURN_CYCLE_TARGET * 100).toFixed(0)}% cycle return`,
            tone: returnPct >= 0 ? "positive" : "negative",
          },
        ],
      },
      risk: {
        title: "Risk Lens",
        subtitle: "Drawdown stress, downside persistence, and control utilization.",
        rows: [
          {
            label: "Max Drawdown",
            value: `${drawdownPct.toFixed(2)}%`,
            hint: "Peak-to-trough equity draw",
            tone: drawdownPct > -8 ? "positive" : "negative",
          },
          {
            label: "Sharpe",
            value: sharpe.toFixed(2),
            hint: "Risk-adjusted return",
            tone: sharpe >= 1 ? "positive" : "negative",
          },
          {
            label: "Open Positions",
            value: String(openPositions),
            hint: "Current risk surface",
          },
          {
            label: "Trades",
            value: String(tradesCount),
            hint: "Execution sample size",
          },
        ],
        bars: [
          {
            label: "DD Budget Use",
            value: clampPct((Math.abs(drawdownPct) / DRAWDOWN_BUDGET_PCT) * 100),
            targetLabel: `${DRAWDOWN_BUDGET_PCT}% portfolio cap`,
            tone: Math.abs(drawdownPct) <= 8 ? "positive" : "negative",
          },
          {
            label: "Risk Buffer",
            value: clampPct(100 - (Math.abs(drawdownPct) / DRAWDOWN_BUDGET_PCT) * 100),
            targetLabel: "Higher is safer",
            tone: Math.abs(drawdownPct) <= 8 ? "positive" : "negative",
          },
          {
            label: "Governance Headroom",
            value: clampPct((sharpe / 2.0) * 100),
            targetLabel: "Sortino/Sharpe guard",
            tone: sharpe >= 1 ? "positive" : "neutral",
          },
        ],
      },
      execution: {
        title: "Execution Lens",
        subtitle: "Fill quality proxies and inventory footprint concentration.",
        rows: [
          {
            label: "Total Trades",
            value: String(tradesCount),
            hint: "Filled orders",
          },
          {
            label: "PnL / Trade",
            value: formatCurrency(pnlPerTrade),
            hint: "Average edge realization",
            tone: pnlPerTrade >= 0 ? "positive" : "negative",
          },
          {
            label: "Avg Position Size",
            value: formatCompactCurrency(avgPositionSize),
            hint: "Equity at risk per position",
          },
          {
            label: "Book Utilization",
            value: formatPct(openPositions / MAX_POSITIONS),
            hint: `Using ${MAX_POSITIONS}-slot max book`,
          },
        ],
        bars: [
          {
            label: "Execution Throughput",
            value: clampPct((tradesCount / TRADE_CYCLE_TARGET) * 100),
            targetLabel: `${TRADE_CYCLE_TARGET} trade cycle`,
            tone: tradesCount >= 20 ? "positive" : "neutral",
          },
          {
            label: "Position Dispersion",
            value: clampPct((openPositions / MAX_POSITIONS) * 100),
            targetLabel: "Concentration monitor",
            tone: openPositions <= Math.floor(MAX_POSITIONS * 0.6) ? "positive" : "negative",
          },
          {
            label: "Edge Capture",
            value: clampPct((pnlPerTrade / EDGE_CAPTURE_TARGET) * 100),
            targetLabel: `$${EDGE_CAPTURE_TARGET}/trade benchmark`,
            tone: pnlPerTrade >= 0 ? "positive" : "negative",
          },
        ],
      },
    };
  }, [capital, dailyPnl, drawdownPct, openPositions, pnlPerTrade, returnPct, sharpe, totalPnl, tradesCount, avgPositionSize, winRate]);

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  const setSort = (key: PositionSortKey) => {
    if (sortKey === key) {
      setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    setSortDirection("desc");
  };

  const dismissAlert = (id: string) => {
    setDismissedAlerts((prev) => ({ ...prev, [id]: true }));
  };

  const fetchMandateHistory = async () => {
    setMandateHistoryLoading(true);
    setMandateHistoryError("");
    try {
      const response = await fetch("/api/v1/mandate/audit?limit=30", { cache: "no-store" });
      const payload = (await response.json().catch(() => ({}))) as { events?: MandateAuditEvent[]; detail?: string };
      if (response.status === 401) {
        handleSessionExpired();
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
  };

  const fetchMandateCalibration = async () => {
    setCalibrationLoading(true);
    setCalibrationError("");
    try {
      const response = await fetch("/api/v1/mandate/calibration?limit=250", { cache: "no-store" });
      const payload = (await response.json().catch(() => ({}))) as MandateCalibrationSnapshot & { detail?: string };
      if (response.status === 401) {
        handleSessionExpired();
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
  };

  const fetchMonthlyRiskReport = async () => {
    setRiskReportLoading(true);
    setRiskReportError("");
    try {
      const response = await fetch("/api/v1/mandate/reports/monthly?lookback=1000", { cache: "no-store" });
      const payload = (await response.json().catch(() => ({}))) as MonthlyModelRiskReport & { detail?: string };
      if (response.status === 401) {
        handleSessionExpired();
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
  };

  const fetchWorkflowList = async () => {
    setWorkflowListLoading(true);
    setWorkflowError("");
    try {
      const response = await fetch("/api/v1/mandate/workflows?limit=50", { cache: "no-store" });
      const payload = (await response.json().catch(() => ([]))) as MandateWorkflowPack[] | { detail?: string };
      if (response.status === 401) {
        handleSessionExpired();
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
  };

  const initiateWorkflowPack = async () => {
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
        handleSessionExpired();
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
  };

  const updateWorkflowStatus = async (workflowId: string, status: MandateWorkflowPack["status"], note: string) => {
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
        handleSessionExpired();
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
  };

  const signoffWorkflow = async (workflowId: string, role: "pm" | "compliance") => {
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
        handleSessionExpired();
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
  };

  const activatePlanFlow = async () => {
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
          handleSessionExpired();
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
          handleSessionExpired();
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
          handleSessionExpired();
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
          handleSessionExpired();
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
  };

  const evaluateMandate = async () => {
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
        handleSessionExpired();
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
  };

  const criticalCount = activeAlerts.filter((a) => a.severity === "critical").length;
  const warningCount = activeAlerts.filter((a) => a.severity === "warning").length;
  const pmSigned = Boolean(workflowPack?.signoffs?.pm?.approved);
  const complianceSigned = Boolean(workflowPack?.signoffs?.compliance?.approved);
  const uspScore = usp?.score ?? 0;
  const copilotSignals = (calibration?.rows?.length ?? 0) + mandateHistory.length + (mandateResult ? 1 : 0);

  const readinessItems = useMemo(
    () => [
      {
        label: "API",
        state: isDisconnected ? "down" : isStale ? "warn" : "ok",
        detail: isDisconnected ? "unreachable" : isStale ? "stale feed" : "healthy",
      },
      {
        label: "Auth",
        state: authExpired ? "down" : "ok",
        detail: authExpired ? "session expired" : "session active",
      },
      {
        label: "USP Engine",
        state: !usp?.engine ? "down" : uspScore < 55 ? "warn" : "ok",
        detail: !usp?.engine ? "missing scorecard" : `score ${uspScore.toFixed(1)}/100`,
      },
      {
        label: "Mandate Copilot",
        state: mandateError || calibrationError ? "warn" : copilotSignals > 0 ? "ok" : "warn",
        detail: mandateError || calibrationError ? "degraded" : `${copilotSignals} signals`,
      },
      {
        label: "Workflow Pack",
        state: workflowError ? "warn" : workflowPack || workflowList.length > 0 ? "ok" : "warn",
        detail: workflowError ? "action required" : workflowPack ? workflowPack.status : `${workflowList.length} packs`,
      },
      {
        label: "Alpaca",
        state: brokerStateToBadge(brokerRuntime.alpaca?.status, brokerRuntime.alpaca?.mode),
        detail: brokerDetail(brokerRuntime.alpaca),
      },
      {
        label: "IBKR",
        state: brokerStateToBadge(brokerRuntime.ibkr?.status, brokerRuntime.ibkr?.mode),
        detail: brokerDetail(brokerRuntime.ibkr),
      },
    ],
    [
      authExpired,
      brokerDetail,
      brokerRuntime.alpaca,
      brokerRuntime.ibkr,
      brokerStateToBadge,
      calibrationError,
      copilotSignals,
      isDisconnected,
      isStale,
      mandateError,
      usp?.engine,
      uspScore,
      workflowError,
      workflowList.length,
      workflowPack,
    ],
  );

  if (authExpired) {
    return (
      <main className="apex-shell min-h-screen px-4 py-6 sm:px-6 lg:px-10">
        <div className="mx-auto w-full max-w-3xl">
          <Alert className="rounded-2xl border-warning/40 bg-warning/10 text-warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Session expired</AlertTitle>
            <AlertDescription>Redirecting to login.</AlertDescription>
          </Alert>
        </div>
      </main>
    );
  }

  return (
    <main className="apex-shell min-h-screen px-4 py-6 sm:px-6 lg:px-10">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 lg:gap-5">
        <AlertsFeed>
          <div className="apex-panel rounded-2xl border border-border/80 px-4 py-3">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex items-center gap-2 text-sm">
                <AlertTriangle className="h-4 w-4 text-primary" />
                <span className="font-semibold text-foreground">Alert Triage</span>
                <span className="rounded-full bg-negative/15 px-2 py-0.5 text-xs font-semibold text-negative">{criticalCount} critical</span>
                <span className="rounded-full bg-warning/15 px-2 py-0.5 text-xs font-semibold text-warning">{warningCount} warning</span>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {(["all", "critical", "warning", "info"] as SeverityFilter[]).map((filter) => (
                  <button
                    key={filter}
                    type="button"
                    aria-pressed={alertFilter === filter}
                    onClick={() => setAlertFilter(filter)}
                    className={`rounded-full px-2.5 py-1.5 text-xs font-semibold transition ${alertFilter === filter
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary text-secondary-foreground hover:bg-secondary/70"
                      }`}
                  >
                    {filter}
                  </button>
                ))}
                <Button size="sm" variant="outline" className="h-9 rounded-full px-3 text-xs" onClick={() => setDrawerOpen((v) => !v)}>
                  {drawerOpen ? "Hide" : "Show"}
                </Button>
              </div>
            </div>

            {drawerOpen ? (
              <div className="mt-3 space-y-2">
                {triageAlerts.length === 0 ? (
                  <p className="rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
                    No active alerts for this filter.
                  </p>
                ) : (
                  triageAlerts.slice(0, 5).map((alert) => (
                    <div key={alert.id} className="flex flex-col gap-2 rounded-lg border border-border/70 bg-background/60 px-3 py-2 sm:flex-row sm:items-start sm:justify-between">
                      <div className="space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className={`rounded-full px-2 py-0.5 text-xs font-semibold uppercase ${severityBadgeClass(alert.severity)}`}>
                            {alert.severity}
                          </span>
                          <span className="text-xs uppercase tracking-wide text-muted-foreground">{alert.source}</span>
                          <p className="text-sm font-semibold text-foreground">{alert.title}</p>
                        </div>
                        <p className="text-xs text-muted-foreground">{alert.detail}</p>
                      </div>
                      <Button size="sm" variant="outline" className="h-9 rounded-full px-3 text-xs" onClick={() => dismissAlert(alert.id)}>
                        Dismiss
                      </Button>
                    </div>
                  ))
                )}
              </div>
            ) : null}
          </div>
        </AlertsFeed>

        <header className="apex-panel apex-fade-up rounded-3xl p-5 sm:p-6">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
            <div className="space-y-2">
              <p className="inline-flex items-center gap-2 rounded-full bg-secondary px-3 py-1 text-xs font-semibold tracking-wide text-secondary-foreground">
                <ShieldCheck className="h-3.5 w-3.5" />
                APEX Trading System
              </p>
              <h1 className="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-sky-400 via-indigo-400 to-emerald-400 drop-shadow-sm pb-1">Apex Trading Terminal</h1>
              <p className="text-sm text-muted-foreground">Real-time execution monitoring.</p>
              <p className="text-xs text-muted-foreground">Desk sync: {nowLabel}</p>
              {aggregatedEquity !== null && (
                <p className="inline-flex items-center gap-1.5 text-xs font-medium text-positive">
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-positive" />
                  Total Equity ({brokerCount} broker{brokerCount !== 1 ? "s" : ""}): {formatCurrency(aggregatedEquity)}
                </p>
              )}
            </div>

            <ControlsPanel>
                            <span
                className={`inline-flex items-center gap-2.5 rounded-full px-3.5 py-1.5 text-xs font-bold uppercase tracking-widest border ${isDisconnected
                  ? "bg-negative/10 text-negative border-negative/30 shadow-[0_0_15px_rgba(239,68,68,0.2)]"
                  : isStale
                    ? "bg-warning/10 text-warning border-warning/30 shadow-[0_0_15px_rgba(245,158,11,0.2)]"
                    : "bg-positive/10 text-positive border-positive/30 shadow-[0_0_15px_rgba(74,222,128,0.2)]"
                  }`}
              >
                <div className="live-ping">
                  <span className={`live-ping-anim ${isDisconnected ? 'hidden' : ''}`}></span>
                  <span className="live-ping-dot"></span>
                </div>
                {isDisconnected ? "System Offline" : isStale ? "Feed Stale" : "Live Engine"}
              </span>
              <Button variant="outline" className="rounded-xl" onClick={() => window.location.reload()}>
                <RefreshCw className="h-4 w-4" />
                Refresh
              </Button>
              <Button
                variant="outline"
                className="rounded-xl"
                onClick={toggleTheme}
                aria-label={
                  !themeMounted
                    ? "Toggle theme"
                    : theme === "dark"
                      ? "Switch to light mode"
                      : "Switch to dark mode"
                }
              >
                {!themeMounted ? <Moon className="h-4 w-4" /> : theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                {!themeMounted ? "Theme" : theme === "dark" ? "Light" : "Dark"}
              </Button>
              {isPublic ? null : (
                <>
                  <Button variant="outline" className="rounded-xl" onClick={handleLogout}>
                    <LogOut className="h-4 w-4" />
                    Logout
                  </Button>
                  <Link
                    href="/settings"
                    className="inline-flex h-9 items-center justify-center rounded-xl border border-border bg-background px-3 text-sm font-medium text-foreground transition hover:bg-secondary/50"
                  >
                    Settings
                  </Link>
                  <Link
                    href="/pricing"
                    className="inline-flex h-9 items-center justify-center rounded-xl border border-border bg-background px-3 text-sm font-medium text-foreground transition hover:bg-secondary/50"
                  >
                    Pricing
                  </Link>
                </>
              )}
            </ControlsPanel>
          </div>
        </header>

        {isDisconnected ? (
          <Alert variant="destructive" className="apex-fade-up rounded-2xl">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Disconnected</AlertTitle>
            <AlertDescription>Backend service is unreachable. Displaying cached data if available.</AlertDescription>
          </Alert>
        ) : null}

        {isStale ? (
          <Alert className="apex-fade-up rounded-2xl border-warning/40 bg-warning/10 text-warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>State stale</AlertTitle>
            <AlertDescription>Backend is reachable, but trading state freshness is below threshold.</AlertDescription>
          </Alert>
        ) : null}

        <section className="apex-panel apex-fade-up rounded-2xl p-4">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-foreground">Function Readiness</h2>
            <span className="text-xs text-muted-foreground">
              Core + USP operating signals {brokerRuntime.active !== "none" ? `• active broker: ${String(brokerRuntime.active).toUpperCase()}` : ""}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-7">
            {readinessItems.map((item) => (
              <div key={item.label} className="rounded-xl border border-border/70 bg-background/60 px-3 py-2">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{item.label}</p>
                  <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase ${readinessClass(item.state as "ok" | "warn" | "down")}`}>
                    {item.state}
                  </span>
                </div>
                <p className="text-xs text-foreground">{item.detail}</p>
              </div>
            ))}
          </div>
        </section>

        <EquityPanel>
          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Capital</p>
              <DollarSign className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-20" /> : formatCurrencyWithCents(capital)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Live equity base</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Daily PnL</p>
              {dailyPnl >= 0 ? <TrendingUp className="h-4 w-4 text-positive" /> : <TrendingDown className="h-4 w-4 text-negative" />}
            </div>
            <p className={`apex-kpi-value mt-2 text-lg font-semibold ${dailyPnl >= 0 ? "text-positive" : "text-negative"}`}>
              {showLoading ? <Skeleton className="h-5 w-20" /> : formatCompactCurrency(dailyPnl)}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">Session contribution</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Sharpe</p>
              <Gauge className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-14" /> : sharpe.toFixed(2)}</p>
            <p className="mt-1 text-xs text-muted-foreground">{`Target ${SHARPE_TARGET.toFixed(2)}+`}</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("risk")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Max Drawdown</p>
              <AlertTriangle className="h-4 w-4 text-primary" />
            </div>
            <p className={`apex-kpi-value mt-2 text-lg font-semibold ${drawdownPct > -8 ? "text-positive" : "text-negative"}`}>
              {showLoading ? <Skeleton className="h-5 w-16" /> : `${drawdownPct.toFixed(2)}%`}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">{`Budget under ${DRAWDOWN_BUDGET_PCT}%`}</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("execution")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Open Positions</p>
              <BarChart3 className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-10" /> : String(openPositions)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Equity symbols only</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("execution")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Total Trades</p>
              <Timer className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-10" /> : String(tradesCount)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Execution sample</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">USP Engine</p>
              <ShieldCheck className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-16" /> : `${(usp?.score ?? 0).toFixed(1)}/100`}</p>
            <p className="mt-1 text-xs capitalize text-muted-foreground">
              {showLoading ? <Skeleton className="h-3 w-20" /> : (usp?.band ?? "stabilize").replaceAll("_", " ")}
            </p>
          </button>
        </EquityPanel>

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[1.25fr_0.75fr]">
          <PositionsTable>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-foreground">KPI Drilldown</h2>
                <p className="text-sm text-muted-foreground">{lensModel[activeLens].subtitle}</p>
              </div>
              <div className="apex-segment" role="tablist" aria-label="KPI lens selector">
                {(["performance", "risk", "execution"] as LensKey[]).map((lens) => (
                  <button
                    key={lens}
                    role="tab"
                    aria-selected={activeLens === lens}
                    type="button"
                    onClick={() => setActiveLens(lens)}
                    className={`apex-segment-button ${activeLens === lens ? "is-active" : ""}`}
                  >
                    {lens}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-5 grid gap-4 lg:grid-cols-2">
              <div className="space-y-3">
                <p className="text-sm font-semibold text-foreground">{lensModel[activeLens].title}</p>
                {lensModel[activeLens].rows.map((row) => (
                  <div key={row.label} className="rounded-xl border border-border/80 bg-background/70 p-3">
                    <div className="flex items-center justify-between gap-4">
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">{row.label}</p>
                      <p className={`apex-kpi-value text-sm font-semibold ${toneClass(row.tone)}`}>{showLoading ? <Skeleton className="h-4 w-16" /> : row.value}</p>
                    </div>
                    <p className="mt-1 text-[11px] text-muted-foreground">{row.hint}</p>
                  </div>
                ))}
              </div>

              <div className="space-y-3">
                <p className="text-sm font-semibold text-foreground">Control Gauges</p>
                {lensModel[activeLens].bars.map((bar) => (
                  <div key={bar.label} className="rounded-xl border border-border/80 bg-background/70 p-3">
                    <div className="mb-2 flex items-center justify-between gap-4 text-xs">
                      <span className="font-medium text-foreground">{bar.label}</span>
                      <span className="text-muted-foreground">{bar.targetLabel}</span>
                    </div>
                    <div className="apex-progress-track">
                      <div className={`apex-progress-fill ${barToneClass(bar.tone)}`} style={{ width: `${showLoading ? 0 : clampPct(bar.value)}%` }} />
                    </div>
                    <p className="apex-kpi-value mt-1 text-[11px] font-medium text-muted-foreground">{showLoading ? <Skeleton className="h-3 w-10" /> : `${clampPct(bar.value).toFixed(0)}%`}</p>
                  </div>
                ))}
              </div>
            </div>
          </PositionsTable>

          <article className="apex-panel apex-fade-up rounded-2xl p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-foreground">Desk Summary</h2>
              <Gauge className="h-4 w-4 text-primary" />
            </div>

            <dl className="mt-4 space-y-2 text-sm">
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Session status</dt>
                <dd className="font-medium text-foreground">
                  {isDisconnected ? "Disconnected" : isStale ? "Stale feed" : "Active"}
                </dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Win rate</dt>
                <dd className="apex-kpi-value font-medium text-foreground">{showLoading ? <Skeleton className="h-4 w-14" /> : formatPct(winRate)}</dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Net PnL</dt>
                <dd className={`apex-kpi-value font-medium ${totalPnl >= 0 ? "text-positive" : "text-negative"}`}>
                  {showLoading ? <Skeleton className="h-4 w-16" /> : formatCompactCurrency(totalPnl)}
                </dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Book utilization</dt>
                <dd className="apex-kpi-value font-medium text-foreground">{showLoading ? <Skeleton className="h-4 w-14" /> : formatPct(openPositions / MAX_POSITIONS)}</dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Alpha retention</dt>
                <dd className="apex-kpi-value font-medium text-foreground">
                  {showLoading ? <Skeleton className="h-4 w-14" /> : `${(usp?.alpha_retention_pct ?? 0).toFixed(1)}%`}
                </dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Data cadence</dt>
                <dd className="font-medium text-foreground">5s polling</dd>
              </div>
            </dl>
          </article>
        </section>

        {/* --- EXPLAINABLE AI SHAP CHART INJECTION --- */}
        <section className="apex-fade-up mb-4 w-full">
          <ExplainableAIChart shapData={activeShapData} symbol="System Aggregate" />
        </section>

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

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[0.95fr_1.05fr]">
          <article className="apex-panel apex-fade-up rounded-2xl p-5">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-foreground">Sleeve Attribution</h2>
              <span className="text-xs text-muted-foreground">30d lookback</span>
            </div>

            <div className="space-y-3">
              {sleeves.length === 0 ? (
                <p className="rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-xs text-muted-foreground">No sleeve attribution yet.</p>
              ) : (
                sleeves.map((sleeve: SleeveAttribution) => (
                  <div key={sleeve.sleeve} className="rounded-xl border border-border/80 bg-background/70 p-3">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-semibold capitalize text-foreground">{formatSleeveLabel(sleeve.sleeve)}</p>
                      <span className="text-xs text-muted-foreground">{sleeve.trades} trades</span>
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                      <div className="rounded-lg border border-border/70 bg-background/60 px-2 py-1.5">
                        <p className="text-muted-foreground">Net alpha</p>
                        <p className={`apex-kpi-value mt-0.5 text-sm font-semibold ${sleeve.net_pnl >= 0 ? "text-positive" : "text-negative"}`}>
                          {formatCurrencyWithCents(sleeve.net_pnl)}
                        </p>
                      </div>
                      <div className="rounded-lg border border-border/70 bg-background/60 px-2 py-1.5">
                        <p className="text-muted-foreground">Execution drag</p>
                        <p className="apex-kpi-value mt-0.5 text-sm font-semibold text-foreground">{formatCurrencyWithCents(sleeve.modeled_execution_drag)}</p>
                      </div>
                      <div className="rounded-lg border border-border/70 bg-background/60 px-2 py-1.5">
                        <p className="text-muted-foreground">Slippage drag</p>
                        <p className="apex-kpi-value mt-0.5 text-sm font-semibold text-foreground">{formatCurrencyWithCents(sleeve.modeled_slippage_drag)}</p>
                      </div>
                      <div className="rounded-lg border border-border/70 bg-background/60 px-2 py-1.5">
                        <p className="text-muted-foreground">Drag / gross</p>
                        <p className="apex-kpi-value mt-0.5 text-sm font-semibold text-foreground">{formatPct(sleeve.execution_drag_pct_of_gross || 0)}</p>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </article>

          <PositionsTable>
            <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-lg font-semibold text-foreground">Position Book (Sortable)</h2>
              <div className="flex flex-wrap items-center gap-2">
                {!isPublic && brokerSources.length > 0 && (
                  <select
                    id="account-selector"
                    value={selectedSourceId ?? ""}
                    onChange={(e) => setSelectedSourceId(e.target.value || null)}
                    className="h-8 rounded-lg border border-border bg-background px-2 text-xs font-medium text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                    aria-label="Filter by account"
                  >
                    <option value="">All Accounts</option>
                    {brokerSources.map((src) => (
                      <option key={src.id} value={src.id}>
                        {src.name} ({src.environment})
                      </option>
                    ))}
                  </select>
                )}
                <span className="text-xs text-muted-foreground">
                  {openPositions} equity + {optionPositions} options = {totalLines} lines
                </span>
              </div>
            </div>

            {notes.map((note) => (
              <p key={note} className="mb-2 rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
                {note}
              </p>
            ))}

            {selectedSourceId && brokerSources.length > 0 && (() => {
              const src = brokerSources.find(s => s.id === selectedSourceId);
              return src ? (
                <p className="mb-2 rounded-lg border border-positive/30 bg-positive/10 px-3 py-2 text-xs font-medium text-positive">
                  Viewing: <strong>{src.name}</strong> ({src.broker_type.toUpperCase()} · {src.environment}) - positions scope active
                </p>
              ) : null;
            })()}

            <div className="max-h-[50vh] overflow-auto rounded-xl border border-border/80">
              <table className="min-w-full text-xs">
                <thead className="sticky top-0 z-10 bg-background/95 backdrop-blur">
                  <tr className="text-left text-muted-foreground">
                    <th className="px-3 py-2 font-semibold">Source</th>
                    {([
                      ["symbol", "Symbol"],
                      ["qty", "Qty"],
                      ["entry", "Entry"],
                      ["current", "Current"],
                      ["pnl", "PnL"],
                      ["pnl_pct", "PnL %"],
                      ["signal_direction", "Signal"],
                    ] as [PositionSortKey, string][]).map(([key, label]) => (
                      <th key={key} className="px-3 py-2 font-semibold" aria-sort={sortKey === key ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}>
                        <button
                          type="button"
                          onClick={() => setSort(key)}
                          className="inline-flex items-center gap-1 hover:text-foreground"
                        >
                          {label}
                          <span className="text-[10px]">{sortIndicator(sortKey === key, sortDirection)}</span>
                        </button>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedPositions.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="px-3 py-6 text-center text-muted-foreground">
                        {showLoading ? "Loading positions..." : "No positions in API state."}
                      </td>
                    </tr>
                  ) : (
                    sortedPositions.map((position) => (
                      <tr key={positionRowKey(position)} className="border-t border-border/60 hover:bg-secondary/30">
                        <td className="px-3 py-2 text-[11px] uppercase text-muted-foreground">
                          {position.broker_type
                            ? `${String(position.broker_type).toUpperCase()}${position.stale ? " (stale)" : ""}`
                            : (position.source_id ? position.source_id.slice(0, 8) : "state")}
                        </td>
                        <td className="px-3 py-2 font-semibold text-foreground">{position.symbol}</td>
                        <td className="apex-kpi-value px-3 py-2 text-foreground">{position.qty}</td>
                        <td className="apex-kpi-value px-3 py-2 text-foreground">{formatCurrencyWithCents(position.entry)}</td>
                        <td className="apex-kpi-value px-3 py-2 text-foreground">{formatCurrencyWithCents(position.current)}</td>
                        <td className={`apex-kpi-value px-3 py-2 font-semibold ${position.pnl >= 0 ? "text-positive" : "text-negative"}`}>
                          {formatCurrencyWithCents(position.pnl)}
                        </td>
                        <td className={`apex-kpi-value px-3 py-2 font-semibold ${position.pnl_pct >= 0 ? "text-positive" : "text-negative"}`}>
                          {position.pnl_pct.toFixed(2)}%
                        </td>
                        <td className="px-3 py-2 uppercase text-muted-foreground">{position.signal_direction || "unknown"}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <div className="mt-4">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-foreground">Derivatives (Options)</h3>
                <span className="text-xs text-muted-foreground">{derivatives.length} option legs</span>
              </div>
              <div className="max-h-[30vh] overflow-auto rounded-xl border border-border/80">
                <table className="min-w-full text-xs">
                  <thead className="sticky top-0 z-10 bg-background/95 backdrop-blur">
                    <tr className="text-left text-muted-foreground">
                      <th className="px-3 py-2 font-semibold">Symbol</th>
                      <th className="px-3 py-2 font-semibold">Expiry</th>
                      <th className="px-3 py-2 font-semibold">Type</th>
                      <th className="px-3 py-2 font-semibold">Strike</th>
                      <th className="px-3 py-2 font-semibold">Qty</th>
                      <th className="px-3 py-2 font-semibold">Avg Cost</th>
                      <th className="px-3 py-2 font-semibold">Side</th>
                    </tr>
                  </thead>
                  <tbody>
                    {derivatives.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="px-3 py-6 text-center text-muted-foreground">
                          {showLoading ? "Loading option legs..." : "No option legs in exported state."}
                        </td>
                      </tr>
                    ) : (
                      derivatives.map((leg: CockpitDerivative) => (
                        <tr key={`${leg.symbol}-${leg.expiry}-${leg.strike}-${leg.right}`} className="border-t border-border/60 hover:bg-secondary/30">
                          <td className="px-3 py-2 font-semibold text-foreground">{leg.symbol}</td>
                          <td className="px-3 py-2 text-foreground">{leg.expiry}</td>
                          <td className="px-3 py-2 text-foreground">{leg.right === "C" ? "CALL" : leg.right === "P" ? "PUT" : leg.right}</td>
                          <td className="apex-kpi-value px-3 py-2 text-foreground">{leg.strike.toFixed(2)}</td>
                          <td className="apex-kpi-value px-3 py-2 text-foreground">{leg.quantity}</td>
                          <td className="apex-kpi-value px-3 py-2 text-foreground">{formatCurrencyWithCents(leg.avg_cost)}</td>
                          <td className="px-3 py-2 text-muted-foreground">{leg.side}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </PositionsTable>
        </section>
      </div>
    </main>
  );
}
