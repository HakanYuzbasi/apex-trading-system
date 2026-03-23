/**
 * lib/api.ts - API client, data fetching hooks, and type definitions
 *
 * Provides SWR-based hooks for cockpit data, metrics, and session-scoped endpoints.
 */
"use client";

import useSWR from "swr";

// ---------------------------------------------------------------------------
// Core API helpers
// ---------------------------------------------------------------------------

const DEFAULT_API_BASE =
  typeof window !== "undefined"
    ? window.location.origin
    : "http://localhost:3000";

export function getApiUrl(path: string): string {
  const base = process.env.NEXT_PUBLIC_API_BASE ?? DEFAULT_API_BASE;
  return `${base}${path}`;
}

export async function apiFetch(
  path: string,
  options: RequestInit & { token?: string | null } = {}
): Promise<Response> {
  const { token, ...init } = options;
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return fetch(getApiUrl(path), { ...init, headers });
}

export async function apiJson<T>(
  path: string,
  options: RequestInit & { token?: string | null } = {}
): Promise<T> {
  const res = await apiFetch(path, options);
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type CockpitAlert = {
  id: string;
  severity: "critical" | "warning" | "info";
  source: string;
  title: string;
  detail: string;
};

export type CockpitPosition = {
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
  replay_url?: string;
};

export type ReplayTimelineEvent = {
  timestamp: string;
  event_type: string;
  symbol: string;
  asset_class: string;
  hash?: string;
  payload: Record<string, unknown>;
};

export type ReplayGovernorPolicySnapshot = {
  policy_key: string;
  policy_id: string;
  version: string;
  asset_class: string;
  regime: string;
  created_at?: string;
  observed_tier?: string;
  tier_controls: Record<string, Record<string, unknown>>;
  metadata: Record<string, unknown>;
  source: string;
};

export type ReplayLiquidationProgress = {
  symbol: string;
  status: string;
  plan_id: string;
  plan_epoch: number;
  planned_reduction_qty: number;
  executed_reduction_qty: number;
  remaining_qty: number;
  progress_pct: number;
  target_reduction_pct: number;
  initial_position_qty: number;
  expected_stress_pnl: number;
  remaining_stress_pnl?: number | null;
  remaining_stress_return?: number | null;
  worst_scenario_id?: string;
  worst_scenario_name?: string;
  breach_event?: ReplayTimelineEvent | null;
  plan_event?: ReplayTimelineEvent | null;
};

export type ReplayPlanAudit = {
  plan_id: string;
  plan_epoch: number;
  started_at?: string | null;
  worst_scenario_id?: string;
  worst_scenario_name?: string;
  candidate_symbols: string[];
  completed_symbols: number;
  in_progress_symbols: number;
  planned_symbols: number;
  breach_event?: ReplayTimelineEvent | null;
  plan_event?: ReplayTimelineEvent | null;
};

export type ReplayChain = {
  chain_id: string;
  symbol: string;
  asset_class: string;
  chain_kind: string;
  started_at: string;
  completed_at?: string | null;
  final_status: string;
  terminal_reason: string;
  signal_event?: ReplayTimelineEvent | null;
  risk_events: ReplayTimelineEvent[];
  order_events: ReplayTimelineEvent[];
  position_events: ReplayTimelineEvent[];
  stress_events: ReplayTimelineEvent[];
  liquidation_progress?: ReplayLiquidationProgress | null;
  governor_policy?: ReplayGovernorPolicySnapshot | null;
};

export type ReplayInspectionResponse = {
  mode: string;
  symbol: string;
  days: number;
  limit: number;
  summary: {
    symbol: string;
    asset_class?: string | null;
    total_events: number;
    total_chains: number;
    blocked_chains: number;
    filled_chains: number;
    open_chains: number;
    stress_liquidation_chains: number;
    latest_event_at?: string | null;
  };
  latest_chain?: ReplayChain | null;
  chains: ReplayChain[];
  raw_events: ReplayTimelineEvent[];
  plan_audit?: ReplayPlanAudit | null;
};

export type CockpitDerivative = {
  symbol: string;
  expiry: string;
  strike: number;
  right: string;
  quantity: number;
  side: string;
  avg_cost: number;
};

export type SleeveAttribution = {
  sleeve: string;
  trades: number;
  net_pnl: number;
  modeled_execution_drag: number;
  modeled_slippage_drag: number;
  execution_drag_pct_of_gross: number;
};

export type SocialAuditEvent = {
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

export type CockpitData = {
  timestamp: string | null;
  positions: CockpitPosition[];
  derivatives: CockpitDerivative[];
  alerts: CockpitAlert[];
  attribution?: {
    sleeves: SleeveAttribution[];
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  status?: any;
  [key: string]: unknown;
};

export type MetricsData = {
  capital: number;
  starting_capital: number;
  daily_pnl: number;
  total_pnl: number;
  max_drawdown: number;
  sharpe_ratio: number;
  win_rate: number;
  open_positions: number;
  total_trades: number;
  [key: string]: unknown;
};

export type SessionInfo = {
  id: string;
  label: string;
  enabled: boolean;
  description: string;
};

export type SessionsResponse = {
  session_mode: string;
  sessions: SessionInfo[];
};

// ---------------------------------------------------------------------------
// Fetchers
// ---------------------------------------------------------------------------

const jsonFetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed: ${res.status}`);
  return res.json();
};

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

export function useCockpitData(isPublic = false) {
  const path = isPublic ? "/api/public/cockpit" : "/api/v1/cockpit";
  const { data, isLoading, isValidating, error } = useSWR<CockpitData>(getApiUrl(path), jsonFetcher, {
    refreshInterval: 5000,
    revalidateOnFocus: true,
  });
  return { data: data ?? null, isLoading, isValidating, isError: !!error, error };
}

export function useMetrics(isPublic = false) {
  const path = isPublic ? "/api/public/metrics" : "/api/v1/metrics";
  const { data, isLoading, isValidating, error } = useSWR<MetricsData>(
    getApiUrl(path),
    jsonFetcher,
    { refreshInterval: 10000 }
  );
  return { metrics: data ?? null, isLoading, isValidating, isError: !!error, error };
}

// ---------------------------------------------------------------------------
// Session-scoped hooks (for dual-session mode)
// ---------------------------------------------------------------------------

export function useSessionStatus(sessionType: string) {
  return useSWR<Record<string, unknown>>(
    getApiUrl(`/api/v1/session/${sessionType}/status`),
    jsonFetcher,
    { refreshInterval: 5000 }
  );
}

export function useSessionPositions(sessionType: string) {
  return useSWR<{ session_type: string; positions: CockpitPosition[] }>(
    getApiUrl(`/api/v1/session/${sessionType}/positions`),
    jsonFetcher,
    { refreshInterval: 5000 }
  );
}

export function useSessionMetrics(sessionType: string) {
  return useSWR<MetricsData & { session_type: string; sharpe_target: number }>(
    getApiUrl(`/api/v1/session/${sessionType}/metrics`),
    jsonFetcher,
    { refreshInterval: 10000 }
  );
}

export function useSessions() {
  return useSWR<SessionsResponse>(
    getApiUrl("/api/v1/sessions"),
    jsonFetcher,
    { refreshInterval: 30000 }
  );
}
