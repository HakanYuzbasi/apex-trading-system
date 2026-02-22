import { NextRequest, NextResponse } from "next/server";
import {
  sanitizeCount,
  sanitizeExecutionMetrics,
  sanitizeMoney,
} from "@/lib/metricGuards";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

type Severity = "critical" | "warning" | "info";

type CockpitAlert = {
  id: string;
  severity: Severity;
  source: string;
  title: string;
  detail: string;
};

type DerivativeRow = {
  symbol: string;
  expiry: string;
  strike: number;
  right: string;
  quantity: number;
  side: string;
  avg_cost: number;
};

type PositionRow = {
  symbol: string;
  qty: number;
  side: string;
  entry: number;
  current: number;
  pnl: number;
  pnl_pct: number;
  signal: number;
  signal_direction: string;
  source_id?: string;
};

type UspScorecard = {
  engine: string;
  score: number;
  band: "institutional_ready" | "improving" | "stabilize";
  sharpe_progress_pct: number;
  drawdown_budget_used_pct: number;
  alpha_retention_pct: number;
  execution_drag_pct_of_gross: number;
};

type SocialAuditEvent = {
  audit_id: string;
  timestamp: string | null;
  asset_class: string;
  regime: string;
  policy_version: string;
  decision_hash: string;
  block_new_entries: boolean;
  gross_exposure_multiplier: number;
  combined_risk_score: number;
  verified_event_probability: number;
  prediction_verification_failures: number;
  verified_event_count: number;
  reasons: string[];
};

type UpstreamResponse = {
  ok: boolean;
  status: number;
  data: unknown;
  transport_error: boolean;
};

type BrokerRuntimeStatus = "live" | "stale" | "configured" | "not_configured";

type BrokerRuntime = {
  broker: "alpaca" | "ibkr";
  status: BrokerRuntimeStatus;
  configured: boolean;
  stale: boolean;
  source_count: number;
  live_source_count: number;
  source_ids: string[];
  total_equity: number;
};

function getApiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

function asNumber(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item ?? "").trim()).filter(Boolean);
}

async function fetchUpstream(path: string, token: string): Promise<UpstreamResponse> {
  try {
    const res = await fetch(`${getApiBase()}${path}`, {
      method: "GET",
      cache: "no-store",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const data = await res.json().catch(() => null);
    return { ok: res.ok, status: res.status, data, transport_error: false };
  } catch {
    return { ok: false, status: 0, data: null, transport_error: true };
  }
}

function isFreshTimestamp(value: unknown, maxAgeSeconds = 180): boolean {
  if (!value) return false;
  const ts = Date.parse(String(value));
  if (!Number.isFinite(ts)) return false;
  const ageMs = Date.now() - ts;
  return ageMs >= 0 && ageMs <= maxAgeSeconds * 1000;
}

function normalizeRight(value: string): string {
  const right = value.trim().toUpperCase();
  if (right === "CALL") return "C";
  if (right === "PUT") return "P";
  if (right === "C" || right === "P") return right;
  return "";
}

function parseHumanExpiry(token: string): string {
  const cleaned = token.trim();
  const match = cleaned.match(/^([A-Za-z]{3})(\d{1,2})'?(\d{2})$/);
  if (!match) return cleaned;
  const monthMap: Record<string, string> = {
    JAN: "01",
    FEB: "02",
    MAR: "03",
    APR: "04",
    MAY: "05",
    JUN: "06",
    JUL: "07",
    AUG: "08",
    SEP: "09",
    OCT: "10",
    NOV: "11",
    DEC: "12",
  };
  const month = monthMap[match[1].toUpperCase()];
  if (!month) return cleaned;
  const day = match[2].padStart(2, "0");
  const year = `20${match[3]}`;
  return `${year}-${month}-${day}`;
}

function parseDerivativeFromPositionLabel(symbol: string, payload: Record<string, unknown>): DerivativeRow | null {
  const normalized = symbol.trim();
  const human = normalized.match(
    /^([A-Z.]+)\s+([A-Za-z]{3}\d{1,2}'?\d{2})\s+(\d+(?:\.\d+)?)\s+(CALL|PUT)$/i,
  );
  if (human) {
    const qty = asNumber(payload.qty ?? payload.quantity, 0);
    const right = normalizeRight(human[4] || "");
    if (!right || qty === 0) return null;
    return {
      symbol: human[1].toUpperCase(),
      expiry: parseHumanExpiry(human[2]),
      strike: asNumber(human[3], 0),
      right,
      quantity: qty,
      side: qty > 0 ? "LONG" : "SHORT",
      avg_cost: asNumber(payload.avg_price ?? payload.avg_cost, 0),
    };
  }

  const occ = normalized.match(/^([A-Z]{1,6})\s*(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/);
  if (occ) {
    const qty = asNumber(payload.qty ?? payload.quantity, 0);
    if (qty === 0) return null;
    return {
      symbol: occ[1],
      expiry: `20${occ[2]}-${occ[3]}-${occ[4]}`,
      strike: asNumber(occ[6], 0) / 1000,
      right: normalizeRight(occ[5]),
      quantity: qty,
      side: qty > 0 ? "LONG" : "SHORT",
      avg_cost: asNumber(payload.avg_price ?? payload.avg_cost, 0),
    };
  }

  return null;
}

function derivativeKey(row: DerivativeRow): string {
  return `${row.symbol}|${row.expiry}|${row.right}|${row.strike}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function normalizePositionRow(row: unknown): PositionRow | null {
  const rec = (row && typeof row === "object" ? row : {}) as Record<string, unknown>;
  const symbol = String(rec.symbol ?? "").trim().toUpperCase();
  if (!symbol) {
    return null;
  }
  const qty = asNumber(rec.qty, 0);
  const sideRaw = String(rec.side ?? (qty < 0 ? "SHORT" : "LONG")).toUpperCase();
  const side = sideRaw === "SHORT" ? "SHORT" : "LONG";
  return {
    symbol,
    qty,
    side,
    entry: asNumber(rec.entry ?? rec.avg_price ?? rec.avg_cost, 0),
    current: asNumber(rec.current ?? rec.current_price, 0),
    pnl: asNumber(rec.pnl ?? rec.unrealized_pl, 0),
    pnl_pct: asNumber(rec.pnl_pct ?? rec.unrealized_plpc, 0),
    signal: asNumber(rec.signal ?? rec.current_signal, 0),
    signal_direction: String(rec.signal_direction ?? "UNKNOWN"),
    source_id: rec.source_id ? String(rec.source_id) : undefined,
  };
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  const token = request.cookies.get("token")?.value;
  if (!token) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }

  const [statusResp, positionsResp, stateResp, socialAuditResp, portfolioPositionsResp, portfolioSourcesResp, portfolioBalanceResp] = await Promise.all([
    fetchUpstream("/status", token),
    fetchUpstream("/positions", token),
    fetchUpstream("/state", token),
    fetchUpstream("/api/v1/social-governor/decisions?limit=40", token),
    fetchUpstream("/portfolio/positions", token),
    fetchUpstream("/portfolio/sources", token),
    fetchUpstream("/portfolio/balance", token),
  ]);

  const statusData = (statusResp.data && typeof statusResp.data === "object" ? statusResp.data : {}) as Record<string, unknown>;
  const stateData = (stateResp.data && typeof stateResp.data === "object" ? stateResp.data : {}) as Record<string, unknown>;
  const stateEndpointPositions = (Array.isArray(positionsResp.data) ? positionsResp.data : [])
    .map((row) => normalizePositionRow(row))
    .filter((row): row is PositionRow => row !== null);
  const portfolioPositions = (Array.isArray(portfolioPositionsResp.data) ? portfolioPositionsResp.data : [])
    .map((row) => normalizePositionRow(row))
    .filter((row): row is PositionRow => row !== null);
  const portfolioSources = (Array.isArray(portfolioSourcesResp.data) ? portfolioSourcesResp.data : []) as Record<string, unknown>[];
  const portfolioBalanceData = (portfolioBalanceResp.data && typeof portfolioBalanceResp.data === "object"
    ? portfolioBalanceResp.data
    : {}) as Record<string, unknown>;
  const portfolioBreakdown = (Array.isArray(portfolioBalanceData.breakdown)
    ? portfolioBalanceData.breakdown
    : []) as Record<string, unknown>[];
  const statePositions = (stateData.positions && typeof stateData.positions === "object"
    ? stateData.positions
    : {}) as Record<string, Record<string, unknown>>;

  const apiReachable = [statusResp, positionsResp, stateResp].some((resp) => !resp.transport_error);
  const hasAuthorizedData = [statusResp, positionsResp, stateResp].some((resp) => resp.ok);
  const allUnauthorized = [statusResp, positionsResp, stateResp].every(
    (resp) => resp.status === 401 || resp.status === 403,
  );
  if (allUnauthorized) {
    return NextResponse.json(
      {
        detail: "Session expired",
        code: "session_expired",
        api_reachable: apiReachable,
      },
      { status: 401 },
    );
  }
  const stateFresh = String(statusData.status || "").toLowerCase() === "online"
    || isFreshTimestamp(stateData.timestamp ?? statusData.timestamp);

  const brokerRuntimeByType = new Map<"alpaca" | "ibkr", BrokerRuntime>();
  for (const broker of ["alpaca", "ibkr"] as const) {
    brokerRuntimeByType.set(broker, {
      broker,
      status: "not_configured",
      configured: false,
      stale: false,
      source_count: 0,
      live_source_count: 0,
      source_ids: [],
      total_equity: 0,
    });
  }
  for (const source of portfolioSources) {
    const brokerRaw = String(source.broker_type ?? "").toLowerCase();
    if (brokerRaw !== "alpaca" && brokerRaw !== "ibkr") {
      continue;
    }
    const current = brokerRuntimeByType.get(brokerRaw as "alpaca" | "ibkr");
    if (!current) continue;
    current.configured = true;
    current.source_count += 1;
    const sourceId = String(source.id ?? "").trim();
    if (sourceId) {
      current.source_ids.push(sourceId);
    }
    if (current.status === "not_configured") {
      current.status = "configured";
    }
  }
  for (const row of portfolioBreakdown) {
    const brokerRaw = String(row.broker ?? "").toLowerCase();
    if (brokerRaw !== "alpaca" && brokerRaw !== "ibkr") {
      continue;
    }
    const current = brokerRuntimeByType.get(brokerRaw as "alpaca" | "ibkr");
    if (!current) continue;
    current.configured = true;
    current.source_count = Math.max(current.source_count, 1);
    const stale = Boolean(row.stale);
    const sourceId = String(row.source_id ?? "").trim();
    if (sourceId && !current.source_ids.includes(sourceId)) {
      current.source_ids.push(sourceId);
    }
    const equity = sanitizeMoney(row.value, 0);
    current.total_equity += equity;
    if (!stale) {
      current.live_source_count += 1;
    } else {
      current.stale = true;
    }
    current.status = !stale ? "live" : current.status === "live" ? "live" : "stale";
  }
  const brokers = Array.from(brokerRuntimeByType.values());
  const liveBrokerCount = brokers.filter((row) => row.status === "live").length;
  const configuredBrokerCount = brokers.filter((row) => row.configured).length;
  const activeBroker = liveBrokerCount > 1
    ? "multi"
    : brokers.find((row) => row.status === "live")?.broker
      ?? brokers.find((row) => row.status === "stale")?.broker
      ?? "none";
  const statusMetrics = sanitizeExecutionMetrics({
    capital: statusData.capital,
    starting_capital: statusData.starting_capital,
    daily_pnl: statusData.daily_pnl,
    total_pnl: statusData.total_pnl,
    max_drawdown: statusData.max_drawdown,
    sharpe_ratio: statusData.sharpe_ratio,
    win_rate: statusData.win_rate,
    open_positions: statusData.open_positions,
    option_positions: statusData.option_positions,
    open_positions_total: statusData.open_positions_total,
    total_trades: statusData.total_trades,
  });
  const combinedCapital = portfolioBalanceResp.ok
    ? sanitizeMoney(portfolioBalanceData.total_equity, statusMetrics.capital)
    : statusMetrics.capital;

  const inferredStateOpenPositions = Math.max(
    statusMetrics.open_positions,
    sanitizeCount(stateData.open_positions, 0),
    stateEndpointPositions.length,
  );
  const usePortfolioPositionsPrimary = portfolioPositions.length > 0;
  const daemonBySymbol = new Map<string, PositionRow>();
  for (const row of stateEndpointPositions) {
    if (!daemonBySymbol.has(row.symbol)) {
      daemonBySymbol.set(row.symbol, row);
    }
  }
  const mergedPortfolioPositions = portfolioPositions.map((row) => {
    const daemonRow = daemonBySymbol.get(row.symbol);
    if (!daemonRow) return row;
    return {
      ...row,
      signal: daemonRow.signal,
      signal_direction: daemonRow.signal_direction,
    };
  });
  const effectivePositions = usePortfolioPositionsPrimary ? mergedPortfolioPositions : stateEndpointPositions;
  const openPositions = Math.max(
    inferredStateOpenPositions,
    effectivePositions.length,
  );
  const derivativesRaw = Array.isArray(stateData.option_positions_detail)
    ? stateData.option_positions_detail
    : [];
  const derivativesMap = new Map<string, DerivativeRow>();
  for (const row of derivativesRaw) {
    const rec = (row && typeof row === "object" ? row : {}) as Record<string, unknown>;
    const right = normalizeRight(String(rec.right || ""));
    const quantity = asNumber(rec.quantity, 0);
    if (!right || quantity === 0) {
      continue;
    }
    const derivative: DerivativeRow = {
      symbol: String(rec.symbol || "").toUpperCase(),
      expiry: String(rec.expiry || ""),
      strike: asNumber(rec.strike, 0),
      right,
      quantity,
      side: quantity > 0 ? "LONG" : "SHORT",
      avg_cost: asNumber(rec.avg_cost, 0),
    };
    derivativesMap.set(derivativeKey(derivative), derivative);
  }

  for (const [symbol, payload] of Object.entries(statePositions)) {
    const derivative = parseDerivativeFromPositionLabel(symbol, payload ?? {});
    if (!derivative) {
      continue;
    }
    const key = derivativeKey(derivative);
    if (derivativesMap.has(key)) {
      continue;
    }
    derivativesMap.set(key, derivative);
  }

  const derivatives = Array.from(derivativesMap.values()).sort((a, b) => {
    return `${a.symbol}|${a.expiry}|${a.right}|${a.strike}`.localeCompare(
      `${b.symbol}|${b.expiry}|${b.right}|${b.strike}`,
    );
  });
  const optionPositions = Math.max(
    statusMetrics.option_positions,
    sanitizeCount(stateData.option_positions, 0),
    derivatives.length,
  );
  const openPositionsTotal = Math.max(
    statusMetrics.open_positions_total,
    sanitizeCount(stateData.open_positions_total, 0),
    openPositions + optionPositions,
  );
  const sharpe = statusMetrics.sharpe_ratio;
  const totalTrades = statusMetrics.trades_count;
  const drawdown = statusMetrics.max_drawdown;
  const absDrawdown = Math.abs(drawdown);
  const normalizedDrawdownPct = absDrawdown > 1 ? absDrawdown : absDrawdown * 100;

  const killSwitch = (stateData.kill_switch && typeof stateData.kill_switch === "object"
    ? stateData.kill_switch
    : {}) as Record<string, unknown>;
  const reconciliation = (stateData.equity_reconciliation && typeof stateData.equity_reconciliation === "object"
    ? stateData.equity_reconciliation
    : {}) as Record<string, unknown>;

  const attributionRaw = (stateData.performance_attribution && typeof stateData.performance_attribution === "object"
    ? stateData.performance_attribution
    : {}) as Record<string, unknown>;
  const bySleeveRaw = (attributionRaw.by_sleeve && typeof attributionRaw.by_sleeve === "object"
    ? attributionRaw.by_sleeve
    : {}) as Record<string, Record<string, unknown>>;
  const attributionGrossPnl = sanitizeMoney(attributionRaw.gross_pnl, 0);
  const attributionNetPnl = sanitizeMoney(attributionRaw.net_pnl, 0);
  const attributionExecutionDrag = sanitizeMoney(attributionRaw.modeled_execution_drag, 0);
  const attributionSlippageDrag = sanitizeMoney(attributionRaw.modeled_slippage_drag, 0);
  const socialAuditData = (socialAuditResp.data && typeof socialAuditResp.data === "object"
    ? socialAuditResp.data
    : {}) as Record<string, unknown>;
  const socialEventsRaw = Array.isArray(socialAuditData.events) ? socialAuditData.events : [];
  const socialEvents = socialEventsRaw
    .map((row): SocialAuditEvent => {
      const rec = (row && typeof row === "object" ? row : {}) as Record<string, unknown>;
      const decision = (rec.decision && typeof rec.decision === "object"
        ? rec.decision
        : {}) as Record<string, unknown>;
      const verifiedEvents = Array.isArray(rec.verified_events) ? rec.verified_events : [];
      return {
        audit_id: String(rec.audit_id || ""),
        timestamp: rec.timestamp ? String(rec.timestamp) : null,
        asset_class: String(rec.asset_class || "").toUpperCase(),
        regime: String(rec.regime || "").toLowerCase(),
        policy_version: String(rec.policy_version || ""),
        decision_hash: String(rec.decision_hash || ""),
        block_new_entries: Boolean(decision.block_new_entries),
        gross_exposure_multiplier: asNumber(decision.gross_exposure_multiplier, 1),
        combined_risk_score: asNumber(decision.combined_risk_score, 0),
        verified_event_probability: asNumber(decision.verified_event_probability, 0),
        prediction_verification_failures: asNumber(decision.prediction_verification_failures, 0),
        verified_event_count: verifiedEvents.length,
        reasons: asStringArray(decision.reasons),
      };
    })
    .filter((row) => Boolean(row.audit_id));
  const socialAuditAvailable = socialAuditResp.ok;
  const socialAuditUnauthorized = socialAuditResp.status === 401 || socialAuditResp.status === 403;
  const socialAuditWarning = socialAuditResp.transport_error
    ? "social_audit_transport_error"
    : (!socialAuditResp.ok && !socialAuditUnauthorized ? `social_audit_http_${socialAuditResp.status}` : "");

  const sleeves = Object.entries(bySleeveRaw).map(([sleeve, row]) => ({
    sleeve,
    trades: sanitizeCount(row.trades),
    net_pnl: sanitizeMoney(row.net_pnl),
    gross_pnl: sanitizeMoney(row.gross_pnl),
    modeled_execution_drag: sanitizeMoney(row.modeled_execution_drag),
    modeled_slippage_drag: sanitizeMoney(row.modeled_slippage_drag),
    execution_drag_pct_of_gross: asNumber(row.execution_drag_pct_of_gross),
    avg_holding_hours: asNumber(row.avg_holding_hours),
  }));

  if (sleeves.length === 0) {
    sleeves.push({
      sleeve: "equities_sleeve",
      trades: sanitizeCount(attributionRaw.closed_trades),
      net_pnl: sanitizeMoney(attributionRaw.net_pnl),
      gross_pnl: sanitizeMoney(attributionRaw.gross_pnl),
      modeled_execution_drag: sanitizeMoney(attributionRaw.modeled_execution_drag),
      modeled_slippage_drag: sanitizeMoney(attributionRaw.modeled_slippage_drag),
      execution_drag_pct_of_gross: 0,
      avg_holding_hours: 0,
    });
  }

  const alerts: CockpitAlert[] = [];
  if (!apiReachable) {
    alerts.push({
      id: "system-offline",
      severity: "critical",
      source: "connectivity",
      title: "Trading API offline",
      detail: "Dashboard is reading stale or unavailable state.",
    });
  }
  if (hasAuthorizedData && apiReachable && !stateFresh) {
    alerts.push({
      id: "state-stale",
      severity: "warning",
      source: "state_freshness",
      title: "State feed stale",
      detail: "API is reachable but trading_state is not updating within freshness window.",
    });
  }
  if (Boolean(killSwitch.active)) {
    alerts.push({
      id: "kill-switch-active",
      severity: "critical",
      source: "kill_switch",
      title: "Kill-switch active",
      detail: String(killSwitch.reason || "Risk circuit breaker is flattening and blocking entries."),
    });
  }
  if (Boolean(reconciliation.block_entries)) {
    alerts.push({
      id: "reconciliation-block",
      severity: "critical",
      source: "reconciliation",
      title: "Reconciliation entry block enabled",
      detail: String(reconciliation.reason || "Equity reconciliation gap is above allowed threshold."),
    });
  }
  if (statusResp.ok && normalizedDrawdownPct >= 8) {
    alerts.push({
      id: "drawdown-warning",
      severity: "warning",
      source: "risk",
      title: "Drawdown pressure elevated",
      detail: `Current drawdown -${normalizedDrawdownPct.toFixed(2)}% is approaching policy limits.`,
    });
  }
  if (statusResp.ok && stateFresh && totalTrades >= 20 && sharpe < 1.0) {
    alerts.push({
      id: "sharpe-warning",
      severity: "warning",
      source: "performance",
      title: "Sharpe below target band",
      detail: `Rolling Sharpe ${sharpe.toFixed(2)} is below the 1.00 caution threshold.`,
    });
  }

  for (const sleeve of sleeves) {
    if (sleeve.trades >= 10 && sleeve.execution_drag_pct_of_gross > 0.35) {
      alerts.push({
        id: `drag-${sleeve.sleeve}`,
        severity: "warning",
        source: "attribution",
        title: `Execution drag spike (${sleeve.sleeve})`,
        detail: `Execution drag is ${(sleeve.execution_drag_pct_of_gross * 100).toFixed(1)}% of gross PnL.`,
      });
    }
  }

  if (positionsResp.ok && openPositions !== effectivePositions.length) {
    alerts.push({
      id: "position-count-mismatch",
      severity: "info",
      source: "state_consistency",
      title: usePortfolioPositionsPrimary ? "Position count mismatch (aggregated)" : "Position count mismatch",
      detail: usePortfolioPositionsPrimary
        ? `Status reports ${openPositions} while /portfolio/positions returned ${effectivePositions.length}.`
        : `Status reports ${openPositions} while positions endpoint returned ${effectivePositions.length}.`,
    });
  }
  if (usePortfolioPositionsPrimary) {
    alerts.push({
      id: "portfolio-position-fallback",
      severity: "info",
      source: "broker_aggregation",
      title: "Using broker aggregated positions",
      detail: "Open positions are sourced from /portfolio/positions (Alpaca + IBKR) with daemon signal enrichment.",
    });
  }

  if (optionPositions > 0) {
    alerts.push({
      id: "derivatives-present",
      severity: "info",
      source: "positions",
      title: "Options positions active",
      detail: `${optionPositions} option contracts are tracked separately from equity symbols in the primary positions table.`,
    });
  }
  if (!socialAuditAvailable && socialAuditWarning) {
    const socialAlertSeverity: Severity = socialAuditResp.transport_error
      ? "warning"
      : socialAuditResp.status >= 500
        ? "info"
        : "warning";
    const socialAlertTitle = socialAuditResp.status >= 500
      ? "Social audit feed degraded"
      : "Social audit feed unavailable";
    alerts.push({
      id: "social-audit-unavailable",
      severity: socialAlertSeverity,
      source: "social_governor",
      title: socialAlertTitle,
      detail: socialAuditResp.status >= 500
        ? `Social-governor audit API returned ${socialAuditResp.status}; core trading telemetry remains available.`
        : `Cockpit could not load social-governor audit feed (${socialAuditWarning}).`,
    });
  }

  const dragTotal = Math.max(0, attributionExecutionDrag + attributionSlippageDrag);
  const alphaRetentionPct = attributionGrossPnl > 0
    ? clamp((attributionNetPnl / attributionGrossPnl) * 100, -200, 200)
    : 0;
  const executionDragPctOfGross = attributionGrossPnl > 0
    ? clamp((dragTotal / attributionGrossPnl) * 100, 0, 400)
    : 0;
  const sharpeProgressPct = clamp((sharpe / 1.5) * 100, 0, 200);
  const drawdownBudgetUsedPct = clamp((Math.abs(normalizedDrawdownPct) / 15) * 100, 0, 200);
  const scoreRaw = (
    0.45 * clamp(sharpeProgressPct, 0, 100)
    + 0.30 * clamp(100 - drawdownBudgetUsedPct, 0, 100)
    + 0.25 * clamp(alphaRetentionPct - executionDragPctOfGross, 0, 100)
  );
  const uspScore = clamp(scoreRaw, 0, 100);
  const uspBand: UspScorecard["band"] = uspScore >= 75
    ? "institutional_ready"
    : uspScore >= 55
      ? "improving"
      : "stabilize";
  const uspScorecard: UspScorecard = {
    engine: "Adaptive Governor + Execution Shield + Attribution Loop",
    score: uspScore,
    band: uspBand,
    sharpe_progress_pct: sharpeProgressPct,
    drawdown_budget_used_pct: drawdownBudgetUsedPct,
    alpha_retention_pct: alphaRetentionPct,
    execution_drag_pct_of_gross: executionDragPctOfGross,
  };

  return NextResponse.json({
    status: {
      online: apiReachable,
      api_reachable: apiReachable,
      state_fresh: stateFresh,
      timestamp: statusData.timestamp ?? null,
      capital: combinedCapital,
      starting_capital: statusMetrics.starting_capital,
      daily_pnl: statusMetrics.daily_pnl,
      total_pnl: statusMetrics.total_pnl,
      max_drawdown: drawdown,
      sharpe_ratio: sharpe,
      win_rate: statusMetrics.win_rate,
      open_positions: openPositions,
      option_positions: optionPositions,
      open_positions_total: openPositionsTotal,
      total_trades: statusMetrics.trades_count,
      brokers,
      active_broker: activeBroker,
      live_broker_count: liveBrokerCount,
      configured_broker_count: configuredBrokerCount,
    },
    positions: effectivePositions,
    derivatives,
    attribution: {
      closed_trades: asNumber(attributionRaw.closed_trades),
      open_positions_tracked: asNumber(attributionRaw.open_positions_tracked),
      gross_pnl: attributionGrossPnl,
      net_pnl: attributionNetPnl,
      commissions: asNumber(attributionRaw.commissions),
      modeled_execution_drag: attributionExecutionDrag,
      modeled_slippage_drag: attributionSlippageDrag,
      sleeves,
    },
    usp: uspScorecard,
    social_audit: {
      available: socialAuditAvailable,
      unauthorized: socialAuditUnauthorized,
      transport_error: socialAuditResp.transport_error,
      status_code: socialAuditResp.status,
      warning: socialAuditWarning || null,
      count: asNumber(socialAuditData.count, socialEvents.length),
      events: socialEvents,
    },
    alerts,
    notes: [
      `Equity positions: ${openPositions}, option contracts: ${optionPositions}, total lines: ${openPositionsTotal}.`,
      `Derivatives table currently includes ${derivatives.length} option legs exported from IBKR.`,
      usePortfolioPositionsPrimary
        ? "Primary equity rows are sourced from /portfolio/positions (Alpaca + IBKR) and enriched with daemon signal fields."
        : "Primary equity rows are sourced from daemon /positions plus state reconciliation.",
      `USP engine score: ${uspScore.toFixed(1)}/100 (${uspBand.replaceAll("_", " ")}).`,
    ],
  });
}
