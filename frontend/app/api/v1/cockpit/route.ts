import { NextRequest, NextResponse } from "next/server";
import {
  sanitizeCount,
  sanitizeExecutionMetrics,
  sanitizeMoney,
} from "@/lib/metricGuards";

export const dynamic = "force-dynamic";
export const revalidate = 0;

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
  broker_type?: "alpaca" | "ibkr";
  security_type?: string;
  expiry?: string;
  strike?: number;
  right?: string;
  stale?: boolean;
  source_status?: string;
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
type BrokerRuntimeMode = "trading" | "idle" | "disabled";

type BrokerRuntime = {
  broker: "alpaca" | "ibkr";
  status: BrokerRuntimeStatus;
  mode: BrokerRuntimeMode;
  configured: boolean;
  stale: boolean;
  source_count: number;
  live_source_count: number;
  source_ids: string[];
  total_equity: number;
  heartbeat_ts: string | null;
  stale_age_seconds: number | null;
};

type BrokerSnapshotCache = {
  updated_at_ms: number;
  brokers: BrokerRuntime[];
  active_broker: "alpaca" | "ibkr" | "multi" | "none";
  live_broker_count: number;
  configured_broker_count: number;
  combined_capital: number;
};

type SocialAuditSnapshotCache = {
  updated_at_ms: number;
  events: SocialAuditEvent[];
  count: number;
  status_code: number;
  warning: string | null;
};

type PositionSnapshotCache = {
  updated_at_ms: number;
  positions: PositionRow[];
  derivatives: DerivativeRow[];
  open_positions: number;
  option_positions: number;
  open_positions_total: number;
};

type PositionParityWatchdogState = {
  updated_at_ms: number;
  mismatch_streak: number;
  last_status_open_positions: number;
  last_table_length: number;
};

const BROKER_RUNTIME_CACHE = new Map<string, BrokerSnapshotCache>();
const BROKER_RUNTIME_CACHE_TTL_MS = Number(process.env.APEX_BROKER_UI_STABILITY_MS || "30000");
const SOCIAL_AUDIT_CACHE = new Map<string, SocialAuditSnapshotCache>();
const SOCIAL_AUDIT_CACHE_TTL_MS = Number(process.env.APEX_SOCIAL_AUDIT_UI_STABILITY_MS || "60000");
const POSITION_SNAPSHOT_CACHE = new Map<string, PositionSnapshotCache>();
const POSITION_SNAPSHOT_CACHE_TTL_MS = Number(process.env.APEX_POSITION_UI_STABILITY_MS || "30000");
const POSITION_PARITY_WATCHDOG = new Map<string, PositionParityWatchdogState>();
const POSITION_PARITY_ALERT_CYCLES = Math.max(1, Number(process.env.APEX_POSITION_PARITY_ALERT_CYCLES || "3"));
const POSITION_PARITY_WATCHDOG_TTL_MS = Number(process.env.APEX_POSITION_PARITY_WATCHDOG_TTL_MS || "300000");

function getApiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

function normalizeBrokerName(value: unknown): "alpaca" | "ibkr" | null {
  const parsed = String(value ?? "").trim().toLowerCase();
  if (parsed === "alpaca" || parsed === "ibkr") return parsed;
  return null;
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

function timestampAgeSeconds(value: unknown): number | null {
  if (!value) return null;
  const ts = Date.parse(String(value));
  if (!Number.isFinite(ts)) return null;
  const ageMs = Date.now() - ts;
  if (!Number.isFinite(ageMs) || ageMs < 0) return 0;
  return Math.floor(ageMs / 1000);
}

function normalizeRight(value: string): string {
  const right = value.trim().toUpperCase();
  if (right === "CALL") return "C";
  if (right === "PUT") return "P";
  if (right === "C" || right === "P") return right;
  return "";
}

function normalizeExpiryToken(value: unknown): string {
  const token = String(value ?? "").trim();
  if (token.length === 8 && /^\d{8}$/.test(token)) {
    return `${token.slice(0, 4)}-${token.slice(4, 6)}-${token.slice(6, 8)}`;
  }
  if (token.length === 6 && /^\d{6}$/.test(token)) {
    return `${token.slice(0, 4)}-${token.slice(4, 6)}-01`;
  }
  return token;
}

function isOptionPosition(row: PositionRow): boolean {
  const securityType = String(row.security_type ?? "").toUpperCase();
  if (securityType.includes("OPT")) {
    return true;
  }
  const right = normalizeRight(String(row.right ?? ""));
  return right === "C" || right === "P";
}

function derivativeFromOptionPosition(row: PositionRow): DerivativeRow | null {
  if (!isOptionPosition(row)) {
    return null;
  }
  const symbolToken = String(row.symbol ?? "").trim().toUpperCase();
  // Aggressive normalization: extract base symbol and remove common suffixes/prefixes
  let baseSymbol = symbolToken.split(/\s+/)[0] || symbolToken;
  baseSymbol = baseSymbol.replace(/^(STK|OPT):/, "").replace(/\.(US|STK|OPT)$/, "").replace(/\/USD$/, "");

  const expiry = normalizeExpiryToken(row.expiry ?? "");
  const right = normalizeRight(String(row.right ?? ""));
  const strike = asNumber(row.strike, 0);

  if (!baseSymbol || !expiry || !right || strike <= 0) {
    return null;
  }
  const quantity = asNumber(row.qty, 0);
  if (quantity === 0) {
    return null;
  }
  return {
    symbol: baseSymbol,
    expiry,
    strike: Number(strike.toFixed(4)), // Normalize precision for keys
    right,
    quantity,
    side: quantity > 0 ? "LONG" : "SHORT",
    avg_cost: asNumber(row.entry, 0),
  };
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
    let baseSymbol = human[1].toUpperCase();
    baseSymbol = baseSymbol.replace(/^(STK|OPT):/, "").replace(/\.(US|STK|OPT)$/, "").replace(/\/USD$/, "");

    return {
      symbol: baseSymbol,
      expiry: parseHumanExpiry(human[2]),
      strike: Number(asNumber(human[3], 0).toFixed(4)),
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
    let baseSymbol = occ[1].toUpperCase();
    baseSymbol = baseSymbol.replace(/^(STK|OPT):/, "").replace(/\.(US|STK|OPT)$/, "").replace(/\/USD$/, "");

    return {
      symbol: baseSymbol,
      expiry: `20${occ[2]}-${occ[3]}-${occ[4]}`,
      strike: Number((asNumber(occ[6], 0) / 1000).toFixed(4)),
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
  const brokerTypeRaw = String(rec.broker_type ?? "").trim().toLowerCase();
  const brokerType = brokerTypeRaw === "alpaca" || brokerTypeRaw === "ibkr"
    ? (brokerTypeRaw as "alpaca" | "ibkr")
    : undefined;
  const securityType = String(rec.security_type ?? rec.sec_type ?? "").trim().toUpperCase();
  const expiry = normalizeExpiryToken(rec.expiry ?? rec.lastTradeDateOrContractMonth ?? "");
  const right = normalizeRight(String(rec.right ?? ""));
  const strike = asNumber(rec.strike, 0);
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
    broker_type: brokerType,
    security_type: securityType || undefined,
    expiry: expiry || undefined,
    strike: strike > 0 ? strike : undefined,
    right: right || undefined,
    stale: Boolean(rec.stale),
    source_status: rec.source_status ? String(rec.source_status) : undefined,
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
  const brokerHeartbeatsRaw = (statusData.broker_heartbeats && typeof statusData.broker_heartbeats === "object"
    ? statusData.broker_heartbeats
    : stateData.broker_heartbeats && typeof stateData.broker_heartbeats === "object"
      ? stateData.broker_heartbeats
      : {}) as Record<string, Record<string, unknown>>;
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

  const cacheKey = token.slice(-24);
  const cachedBrokerSnapshot = BROKER_RUNTIME_CACHE.get(cacheKey);
  const cachedBrokerSnapshotFresh = Boolean(
    cachedBrokerSnapshot && (Date.now() - cachedBrokerSnapshot.updated_at_ms) <= BROKER_RUNTIME_CACHE_TTL_MS,
  );
  const cachedSocialAuditSnapshot = SOCIAL_AUDIT_CACHE.get(cacheKey);
  const cachedSocialAuditSnapshotFresh = Boolean(
    cachedSocialAuditSnapshot
    && (Date.now() - cachedSocialAuditSnapshot.updated_at_ms) <= SOCIAL_AUDIT_CACHE_TTL_MS,
  );
  const cachedPositionSnapshot = POSITION_SNAPSHOT_CACHE.get(cacheKey);
  const cachedPositionSnapshotFresh = Boolean(
    cachedPositionSnapshot
    && (Date.now() - cachedPositionSnapshot.updated_at_ms) <= POSITION_SNAPSHOT_CACHE_TTL_MS,
  );
  for (const [key, row] of POSITION_PARITY_WATCHDOG.entries()) {
    if ((Date.now() - row.updated_at_ms) > POSITION_PARITY_WATCHDOG_TTL_MS) {
      POSITION_PARITY_WATCHDOG.delete(key);
    }
  }

  const brokerRuntimeByType = new Map<"alpaca" | "ibkr", BrokerRuntime>();
  for (const broker of ["alpaca", "ibkr"] as const) {
    brokerRuntimeByType.set(broker, {
      broker,
      status: "not_configured",
      mode: "disabled",
      configured: false,
      stale: false,
      source_count: 0,
      live_source_count: 0,
      source_ids: [],
      total_equity: 0,
      heartbeat_ts: null,
      stale_age_seconds: null,
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
    current.mode = "idle";
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
    const asOf = row.as_of ? String(row.as_of) : null;
    if (asOf) {
      const currentTs = current.heartbeat_ts ? Date.parse(current.heartbeat_ts) : NaN;
      const incomingTs = Date.parse(asOf);
      if (!Number.isFinite(currentTs) || (Number.isFinite(incomingTs) && incomingTs > currentTs)) {
        current.heartbeat_ts = asOf;
      }
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
  for (const row of portfolioPositions) {
    if (row.broker_type !== "alpaca" && row.broker_type !== "ibkr") {
      continue;
    }
    const current = brokerRuntimeByType.get(row.broker_type);
    if (!current) continue;
    current.configured = true;
    current.source_count = Math.max(current.source_count, 1);
    if (row.source_id && !current.source_ids.includes(row.source_id)) {
      current.source_ids.push(row.source_id);
    }
    if (!row.stale) {
      current.live_source_count = Math.max(current.live_source_count, 1);
      current.status = "live";
    } else {
      current.stale = true;
      if (current.status === "not_configured") {
        current.status = "stale";
      }
    }
  }
  for (const broker of ["alpaca", "ibkr"] as const) {
    const heartbeat = brokerHeartbeatsRaw[broker];
    if (!heartbeat || typeof heartbeat !== "object") continue;
    const current = brokerRuntimeByType.get(broker);
    if (!current) continue;
    const successTs = heartbeat.last_success_ts ? String(heartbeat.last_success_ts) : null;
    const errorTs = heartbeat.last_error_ts ? String(heartbeat.last_error_ts) : null;
    if (successTs) {
      const currentTs = current.heartbeat_ts ? Date.parse(current.heartbeat_ts) : NaN;
      const successMs = Date.parse(successTs);
      if (!Number.isFinite(currentTs) || (Number.isFinite(successMs) && successMs > currentTs)) {
        current.heartbeat_ts = successTs;
      }
    }
    if (!successTs && errorTs && !current.heartbeat_ts) {
      current.heartbeat_ts = errorTs;
    }
    if (heartbeat.healthy === false && current.status === "live") {
      current.status = "stale";
      current.stale = true;
    }
  }
  for (const row of brokerRuntimeByType.values()) {
    row.stale_age_seconds = timestampAgeSeconds(row.heartbeat_ts);
  }
  const preferredActiveBroker = normalizeBrokerName(
    statusData.primary_execution_broker
    || process.env.APEX_PRIMARY_EXECUTION_BROKER
    || process.env.APEX_ACTIVE_TRADING_BROKER
    || "alpaca",
  );
  const brokerModeHint = String(statusData.broker_mode ?? "").trim().toLowerCase();
  for (const broker of ["alpaca", "ibkr"] as const) {
    const impliedByMode = brokerModeHint === "both" || brokerModeHint === broker;
    if (!impliedByMode) continue;
    const runtime = brokerRuntimeByType.get(broker);
    if (!runtime || runtime.configured) continue;
    runtime.configured = true;
    runtime.source_count = Math.max(runtime.source_count, 1);
    runtime.status = "configured";
  }
  if (preferredActiveBroker) {
    const preferredRuntime = brokerRuntimeByType.get(preferredActiveBroker);
    if (preferredRuntime && !preferredRuntime.configured) {
      preferredRuntime.configured = true;
      preferredRuntime.source_count = Math.max(preferredRuntime.source_count, 1);
      preferredRuntime.status = "configured";
    }
  }
  let brokers = Array.from(brokerRuntimeByType.values());
  let liveBrokerCount = brokers.filter((row) => row.status === "live").length;
  let configuredBrokerCount = brokers.filter((row) => row.configured).length;
  let activeBroker: "alpaca" | "ibkr" | "multi" | "none" =
    liveBrokerCount > 1
      ? "multi"
      : brokers.find((row) => row.status === "live")?.broker
      ?? brokers.find((row) => row.status === "stale")?.broker
      ?? "none";

  if (preferredActiveBroker) {
    const preferredConfigured = brokers.some(
      (row) => row.broker === preferredActiveBroker && row.configured,
    );
    if (preferredConfigured || activeBroker === "none") {
      activeBroker = preferredActiveBroker;
    }
  }

  brokers = brokers.map((row) => {
    if (!row.configured) {
      return { ...row, mode: "disabled" as BrokerRuntimeMode };
    }
    // In multi-broker mode, all configured live/stale brokers are actively trading
    if (activeBroker === "multi" && (row.status === "live" || row.status === "stale" || row.status === "configured")) {
      return { ...row, mode: "trading" as BrokerRuntimeMode, stale: row.status === "stale" };
    }
    if (activeBroker !== "none" && activeBroker !== "multi" && row.broker === activeBroker) {
      return { ...row, mode: "trading" as BrokerRuntimeMode };
    }
    // Non-active broker is intentionally idle/read-only for metrics + positions.
    return {
      ...row,
      mode: "idle" as BrokerRuntimeMode,
      status: row.status === "not_configured" ? "configured" : row.status,
    };
  });
  liveBrokerCount = brokers.filter((row) => row.status === "live" || row.mode === "trading").length;
  configuredBrokerCount = brokers.filter((row) => row.configured).length;
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
  const dailyPnlSource = String(statusData.daily_pnl_source ?? "").trim().toLowerCase();
  const statusDailyPnlRealized = sanitizeMoney(statusData.daily_pnl_realized, statusMetrics.daily_pnl);
  const statusDailyPnlByBrokerRaw = (statusData.daily_pnl_by_broker && typeof statusData.daily_pnl_by_broker === "object"
    ? statusData.daily_pnl_by_broker
    : {}) as Record<string, unknown>;
  const statusDailyPnlByBroker = {
    ibkr: sanitizeMoney(statusDailyPnlByBrokerRaw.ibkr, 0),
    alpaca: sanitizeMoney(statusDailyPnlByBrokerRaw.alpaca, 0),
  };
  let combinedCapital = portfolioBalanceResp.ok
    ? sanitizeMoney(portfolioBalanceData.total_equity, statusMetrics.capital)
    : statusMetrics.capital;
  const upstreamBrokerSignalDegraded = portfolioBalanceResp.transport_error
    || portfolioSourcesResp.transport_error
    || portfolioPositionsResp.transport_error
    || portfolioBalanceResp.status >= 500
    || portfolioSourcesResp.status >= 500
    || portfolioPositionsResp.status >= 500;

  if (upstreamBrokerSignalDegraded && cachedBrokerSnapshotFresh && cachedBrokerSnapshot) {
    const previousByBroker = new Map(cachedBrokerSnapshot.brokers.map((row) => [row.broker, row]));
    brokers = brokers.map((row) => {
      const previous = previousByBroker.get(row.broker);
      if (!previous) {
        return row;
      }
      if (previous.status === "live" && row.status !== "live" && row.configured) {
        return {
          ...row,
          status: "live" as BrokerRuntimeStatus,
          stale: false,
          live_source_count: Math.max(1, row.live_source_count),
        };
      }
      return row;
    });
    liveBrokerCount = brokers.filter((row) => row.status === "live").length;
    configuredBrokerCount = brokers.filter((row) => row.configured).length;
    if (liveBrokerCount > 1) {
      activeBroker = "multi";
    } else if (liveBrokerCount === 1) {
      activeBroker = brokers.find((row) => row.status === "live")?.broker ?? activeBroker;
    }
  }

  // Hysteresis: keep the actively-trading broker "live" for a short window to avoid
  // readiness flapping on transient source refresh misses.
  if (cachedBrokerSnapshotFresh && cachedBrokerSnapshot) {
    const previousByBroker = new Map(cachedBrokerSnapshot.brokers.map((row) => [row.broker, row]));
    const targetActiveBroker = activeBroker !== "none" && activeBroker !== "multi"
      ? activeBroker
      : cachedBrokerSnapshot.active_broker;
    if (targetActiveBroker !== "none" && targetActiveBroker !== "multi") {
      const previous = previousByBroker.get(targetActiveBroker);
      brokers = brokers.map((row) => {
        if (row.broker !== targetActiveBroker) return row;
        if (!row.configured || row.mode !== "trading") return row;
        if (row.status === "live") return row;
        if (previous?.status === "live") {
          return {
            ...row,
            status: "live" as BrokerRuntimeStatus,
            stale: false,
            live_source_count: Math.max(1, row.live_source_count),
          };
        }
        return row;
      });
      liveBrokerCount = brokers.filter((row) => row.status === "live").length;
      configuredBrokerCount = brokers.filter((row) => row.configured).length;
      if (liveBrokerCount > 1) {
        activeBroker = "multi";
      } else if (liveBrokerCount === 1) {
        activeBroker = brokers.find((row) => row.status === "live")?.broker ?? activeBroker;
      } else if (targetActiveBroker) {
        activeBroker = targetActiveBroker;
      }
    }
  }

  if ((!portfolioBalanceResp.ok || configuredBrokerCount === 0) && cachedBrokerSnapshotFresh && cachedBrokerSnapshot) {
    brokers = cachedBrokerSnapshot.brokers;
    activeBroker = cachedBrokerSnapshot.active_broker;
    liveBrokerCount = cachedBrokerSnapshot.live_broker_count;
    configuredBrokerCount = cachedBrokerSnapshot.configured_broker_count;
    combinedCapital = cachedBrokerSnapshot.combined_capital;
  } else if (configuredBrokerCount > 0 || portfolioBalanceResp.ok) {
    BROKER_RUNTIME_CACHE.set(cacheKey, {
      updated_at_ms: Date.now(),
      brokers,
      active_broker: activeBroker,
      live_broker_count: liveBrokerCount,
      configured_broker_count: configuredBrokerCount,
      combined_capital: combinedCapital,
    });
  }
  brokers = brokers.map((row) => ({
    ...row,
    stale_age_seconds: timestampAgeSeconds(row.heartbeat_ts),
  }));

  const stateEndpointEquityPositions = stateEndpointPositions.filter((row) => !isOptionPosition(row));
  const stateEndpointOptionPositions = stateEndpointPositions.filter((row) => isOptionPosition(row));
  const portfolioEquityPositions = portfolioPositions.filter((row) => !isOptionPosition(row));
  const portfolioOptionPositions = portfolioPositions.filter((row) => isOptionPosition(row));
  const inferredStateOpenPositions = Math.max(
    statusMetrics.open_positions,
    sanitizeCount(stateData.open_positions, 0),
    stateEndpointEquityPositions.length,
  );
  const usePortfolioPositionsPrimary = portfolioEquityPositions.length > 0;
  const daemonBySymbol = new Map<string, PositionRow>();
  for (const row of stateEndpointEquityPositions) {
    if (!daemonBySymbol.has(row.symbol)) {
      daemonBySymbol.set(row.symbol, row);
    }
  }
  const mergedPortfolioPositions = portfolioEquityPositions.map((row) => {
    const daemonRow = daemonBySymbol.get(row.symbol);
    if (!daemonRow) return row;
    return {
      ...row,
      signal: daemonRow.signal,
      signal_direction: daemonRow.signal_direction,
    };
  });
  const effectivePositionsCandidate = usePortfolioPositionsPrimary ? mergedPortfolioPositions : stateEndpointEquityPositions;
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
  for (const optionRow of [...portfolioOptionPositions, ...stateEndpointOptionPositions]) {
    const derivative = derivativeFromOptionPosition(optionRow);
    if (!derivative) {
      continue;
    }
    const key = derivativeKey(derivative);
    if (derivativesMap.has(key)) {
      continue;
    }
    derivativesMap.set(key, derivative);
  }

  const derivativesCandidate = Array.from(derivativesMap.values()).sort((a, b) => {
    return `${a.symbol}|${a.expiry}|${a.right}|${a.strike}`.localeCompare(
      `${b.symbol}|${b.expiry}|${b.right}|${b.strike}`,
    );
  });
  const upstreamPositionsDegraded = portfolioPositionsResp.transport_error
    || positionsResp.transport_error
    || stateResp.transport_error
    || portfolioPositionsResp.status >= 500
    || positionsResp.status >= 500
    || stateResp.status >= 500;
  const confirmedFlatState = stateFresh
    && !upstreamPositionsDegraded
    && inferredStateOpenPositions === 0
    && sanitizeCount(stateData.option_positions, 0) === 0
    && effectivePositionsCandidate.length === 0
    && derivativesCandidate.length === 0;
  let effectivePositions = effectivePositionsCandidate;
  let derivatives = derivativesCandidate;
  let usingPositionCache = false;

  if (
    effectivePositionsCandidate.length === 0
    && derivativesCandidate.length === 0
    && cachedPositionSnapshotFresh
    && cachedPositionSnapshot
    && (cachedPositionSnapshot.positions.length > 0 || cachedPositionSnapshot.derivatives.length > 0)
    && !confirmedFlatState
    && (upstreamPositionsDegraded || !stateFresh || inferredStateOpenPositions > 0 || statusMetrics.open_positions > 0)
  ) {
    effectivePositions = cachedPositionSnapshot.positions;
    derivatives = cachedPositionSnapshot.derivatives;
    usingPositionCache = true;
  } else if (effectivePositionsCandidate.length > 0 || derivativesCandidate.length > 0) {
    const optionPositionsSnapshot = Math.max(
      statusMetrics.option_positions,
      sanitizeCount(stateData.option_positions, 0),
      derivativesCandidate.length,
    );
    const openPositionsSnapshot = Math.max(
      inferredStateOpenPositions,
      effectivePositionsCandidate.length,
    );
    POSITION_SNAPSHOT_CACHE.set(cacheKey, {
      updated_at_ms: Date.now(),
      positions: effectivePositionsCandidate,
      derivatives: derivativesCandidate,
      open_positions: openPositionsSnapshot,
      option_positions: optionPositionsSnapshot,
      open_positions_total: Math.max(
        statusMetrics.open_positions_total,
        sanitizeCount(stateData.open_positions_total, 0),
        openPositionsSnapshot + optionPositionsSnapshot,
      ),
    });
  } else if (confirmedFlatState) {
    POSITION_SNAPSHOT_CACHE.delete(cacheKey);
  }

  const cachedOpenPositions = usingPositionCache && cachedPositionSnapshot
    ? cachedPositionSnapshot.open_positions
    : 0;
  const cachedOptionPositions = usingPositionCache && cachedPositionSnapshot
    ? cachedPositionSnapshot.option_positions
    : 0;
  const cachedOpenPositionsTotal = usingPositionCache && cachedPositionSnapshot
    ? cachedPositionSnapshot.open_positions_total
    : 0;
  const openPositions = Math.max(
    inferredStateOpenPositions,
    effectivePositions.length,
    cachedOpenPositions,
  );
  const optionPositions = Math.max(
    statusMetrics.option_positions,
    sanitizeCount(stateData.option_positions, 0),
    derivatives.length,
    cachedOptionPositions,
  );
  const openPositionsTotal = Math.max(
    statusMetrics.open_positions_total,
    sanitizeCount(stateData.open_positions_total, 0),
    openPositions + optionPositions,
    cachedOpenPositionsTotal,
  );
  const inferredPortfolioPnl = effectivePositions.reduce(
    (sum, row) => sum + sanitizeMoney(row.pnl, 0),
    0,
  );
  const inferredCombinedPnl = (usePortfolioPositionsPrimary ? portfolioPositions : [...stateEndpointPositions, ...portfolioPositions]).reduce(
    (sum, row) => sum + sanitizeMoney(row.pnl, 0),
    0,
  );
  const resolvedTotalPnl = Math.abs(statusMetrics.total_pnl) > 1e-9
    ? statusMetrics.total_pnl
    : (Math.abs(inferredCombinedPnl) > 1e-9 ? inferredCombinedPnl : inferredPortfolioPnl);
  const resolvedDailyPnl = dailyPnlSource === "broker_fills"
    ? statusDailyPnlRealized
    : (Math.abs(statusMetrics.daily_pnl) > 1e-9
      ? statusMetrics.daily_pnl
      : (Math.abs(inferredCombinedPnl) > 1e-9 ? inferredCombinedPnl : inferredPortfolioPnl));
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
  let socialEvents = socialEventsRaw
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
  let socialAuditAvailable = socialAuditResp.ok;
  const socialAuditUnauthorized = socialAuditResp.status === 401 || socialAuditResp.status === 403;
  let socialAuditWarning = socialAuditResp.transport_error
    ? "social_audit_transport_error"
    : (!socialAuditResp.ok && !socialAuditUnauthorized ? `social_audit_http_${socialAuditResp.status}` : "");
  let socialAuditCount = asNumber(socialAuditData.count, socialEvents.length);
  let socialAuditStatusCode = socialAuditResp.status;
  let socialAuditFromCache = false;

  const shouldKeepCachedEvents = cachedSocialAuditSnapshotFresh
    && cachedSocialAuditSnapshot
    && cachedSocialAuditSnapshot.events.length > 0
    && socialEvents.length === 0;

  if (socialAuditResp.ok) {
    if (shouldKeepCachedEvents && cachedSocialAuditSnapshot) {
      socialEvents = cachedSocialAuditSnapshot.events;
      socialAuditCount = cachedSocialAuditSnapshot.count;
      socialAuditWarning = "social_audit_cached_empty_payload";
      socialAuditFromCache = true;
    } else {
      SOCIAL_AUDIT_CACHE.set(cacheKey, {
        updated_at_ms: Date.now(),
        events: socialEvents,
        count: socialAuditCount,
        status_code: socialAuditResp.status || 200,
        warning: socialAuditWarning || null,
      });
    }
  } else if (cachedSocialAuditSnapshotFresh && cachedSocialAuditSnapshot) {
    socialAuditAvailable = true;
    socialAuditFromCache = true;
    socialEvents = cachedSocialAuditSnapshot.events;
    socialAuditCount = cachedSocialAuditSnapshot.count;
    socialAuditStatusCode = cachedSocialAuditSnapshot.status_code || 200;
    socialAuditWarning = socialAuditResp.transport_error
      ? "social_audit_cached_transport_error"
      : `social_audit_cached_http_${socialAuditResp.status}`;
  }
  const socialShock = (stateData.social_shock && typeof stateData.social_shock === "object"
    ? stateData.social_shock
    : {}) as Record<string, unknown>;
  const socialShockDecisions = Array.isArray(socialShock.decisions) ? socialShock.decisions : [];
  if (socialEvents.length === 0 && socialShockDecisions.length > 0) {
    const fallbackEvents = socialShockDecisions
      .map((row): SocialAuditEvent => {
        const rec = (row && typeof row === "object" ? row : {}) as Record<string, unknown>;
        return {
          audit_id: String(rec.audit_id || `state-${String(rec.asset_class || "asset").toLowerCase()}-${String(rec.regime || "default").toLowerCase()}`),
          timestamp: rec.timestamp ? String(rec.timestamp) : null,
          asset_class: String(rec.asset_class || "").toUpperCase(),
          regime: String(rec.regime || "").toLowerCase(),
          policy_version: String(rec.policy_version || ""),
          decision_hash: String(rec.decision_hash || ""),
          block_new_entries: Boolean(rec.block_new_entries),
          gross_exposure_multiplier: asNumber(rec.gross_exposure_multiplier, 1),
          combined_risk_score: asNumber(rec.combined_risk_score ?? rec.social_risk_score, 0),
          verified_event_probability: asNumber(rec.verified_event_probability, 0),
          prediction_verification_failures: asNumber(rec.prediction_verification_failures, 0),
          verified_event_count: asNumber(rec.verified_event_count, 0),
          reasons: asStringArray(rec.reasons),
        };
      })
      .filter((row) => Boolean(row.audit_id));
    if (fallbackEvents.length > 0) {
      socialEvents = fallbackEvents;
      socialAuditCount = fallbackEvents.length;
      socialAuditAvailable = true;
      socialAuditFromCache = true;
      socialAuditStatusCode = socialAuditStatusCode || 200;
      socialAuditWarning = socialAuditWarning
        ? `${socialAuditWarning}|social_audit_state_fallback`
        : "social_audit_state_fallback";
      SOCIAL_AUDIT_CACHE.set(cacheKey, {
        updated_at_ms: Date.now(),
        events: fallbackEvents,
        count: fallbackEvents.length,
        status_code: socialAuditStatusCode || 200,
        warning: socialAuditWarning || null,
      });
    }
  }

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
    if (effectivePositions.length > 0) {
      const byInferredSleeve = new Map<string, {
        trades: number;
        net_pnl: number;
        gross_pnl: number;
      }>();
      for (const row of effectivePositions) {
        const sleeveKey = row.broker_type ? `${row.broker_type}_sleeve` : "equities_sleeve";
        const current = byInferredSleeve.get(sleeveKey) || {
          trades: 0,
          net_pnl: 0,
          gross_pnl: 0,
        };
        current.trades += 1;
        current.net_pnl += sanitizeMoney(row.pnl, 0);
        current.gross_pnl += Math.abs(sanitizeMoney(row.pnl, 0));
        byInferredSleeve.set(sleeveKey, current);
      }
      for (const [sleeve, stats] of byInferredSleeve.entries()) {
        sleeves.push({
          sleeve,
          trades: stats.trades,
          net_pnl: stats.net_pnl,
          gross_pnl: stats.gross_pnl,
          modeled_execution_drag: 0,
          modeled_slippage_drag: 0,
          execution_drag_pct_of_gross: 0,
          avg_holding_hours: 0,
        });
      }
    } else {
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

  const nowMs = Date.now();
  const parityMismatch = positionsResp.ok && openPositions !== effectivePositions.length;
  const priorParity = POSITION_PARITY_WATCHDOG.get(cacheKey);
  const parityExpired = Boolean(priorParity && (nowMs - priorParity.updated_at_ms) > POSITION_PARITY_WATCHDOG_TTL_MS);
  const mismatchStreak = parityMismatch
    ? ((parityExpired || !priorParity) ? 1 : priorParity.mismatch_streak + 1)
    : 0;
  POSITION_PARITY_WATCHDOG.set(cacheKey, {
    updated_at_ms: nowMs,
    mismatch_streak: mismatchStreak,
    last_status_open_positions: openPositions,
    last_table_length: effectivePositions.length,
  });
  if (parityMismatch && mismatchStreak >= POSITION_PARITY_ALERT_CYCLES) {
    alerts.push({
      id: "position-parity-watchdog",
      severity: "warning",
      source: "state_consistency",
      title: "Position parity watchdog triggered",
      detail: `status.open_positions=${openPositions} vs table=${effectivePositions.length} for ${mismatchStreak} consecutive cycles.`,
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
  if (usingPositionCache) {
    alerts.push({
      id: "position-cache-fallback",
      severity: "info",
      source: "state_consistency",
      title: "Using cached position snapshot",
      detail: "Position rows are held from the last good snapshot while upstream feeds recover.",
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
      : socialAuditStatusCode >= 500
        ? "info"
        : "warning";
    const socialAlertTitle = socialAuditStatusCode >= 500
      ? "Social audit feed degraded"
      : "Social audit feed unavailable";
    alerts.push({
      id: "social-audit-unavailable",
      severity: socialAlertSeverity,
      source: "social_governor",
      title: socialAlertTitle,
      detail: socialAuditStatusCode >= 500
        ? `Social-governor audit API returned ${socialAuditStatusCode}; core trading telemetry remains available.`
        : `Cockpit could not load social-governor audit feed (${socialAuditWarning}).`,
    });
  } else if (socialAuditFromCache && socialAuditWarning) {
    alerts.push({
      id: "social-audit-cached",
      severity: "info",
      source: "social_governor",
      title: "Social audit using cached snapshot",
      detail: `Live social-governor feed is degraded (${socialAuditWarning}); showing last good snapshot.`,
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
      daily_pnl: resolvedDailyPnl,
      daily_pnl_realized: statusDailyPnlRealized,
      daily_pnl_source: dailyPnlSource === "broker_fills" ? "broker_fills" : "inferred",
      daily_pnl_by_broker: statusDailyPnlByBroker,
      total_pnl: resolvedTotalPnl,
      max_drawdown: drawdown,
      sharpe_ratio: sharpe,
      win_rate: statusMetrics.win_rate,
      open_positions: openPositions,
      option_positions: optionPositions,
      open_positions_total: openPositionsTotal,
      total_trades: statusMetrics.trades_count,
      broker_mode: brokerModeHint || "both",
      primary_execution_broker: preferredActiveBroker ?? "alpaca",
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
      status_code: socialAuditStatusCode,
      warning: socialAuditWarning || null,
      count: socialAuditCount,
      events: socialEvents,
      cached: socialAuditFromCache,
    },
    alerts,
    notes: [
      `Equity positions: ${openPositions}, option contracts: ${optionPositions}, total lines: ${openPositionsTotal}.`,
      `Derivatives table currently includes ${derivatives.length} option legs exported from IBKR.`,
      activeBroker !== "none" && activeBroker !== "multi"
        ? `Execution routing is pinned to ${String(activeBroker).toUpperCase()} (other configured brokers stay idle/read-only).`
        : "Execution routing has no single active broker pin.",
      usePortfolioPositionsPrimary
        ? "Primary equity rows are sourced from /portfolio/positions (Alpaca + IBKR) and enriched with daemon signal fields."
        : "Primary equity rows are sourced from daemon /positions plus state reconciliation.",
      usingPositionCache
        ? "Position and derivatives tables are temporarily served from cached snapshot due transient upstream degradation."
        : "Position and derivatives tables are currently served from live upstream snapshots.",
      socialAuditFromCache
        ? "Social Governor audit is currently served from cached snapshot due upstream degradation."
        : "Social Governor audit feed is live.",
      dailyPnlSource === "broker_fills"
        ? "Daily PnL is sourced from broker fill-realized ledger (not unrealized inference)."
        : "Daily PnL may include inferred unrealized fallback when broker fill ledger is unavailable.",
      `USP engine score: ${uspScore.toFixed(1)}/100 (${uspBand.replaceAll("_", " ")}).`,
    ],
  });
}
