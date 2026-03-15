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

export type CockpitData = {
  timestamp: string | null;
  positions: CockpitPosition[];
  derivatives: CockpitDerivative[];
  alerts: CockpitAlert[];
  attribution?: {
    sleeves: SleeveAttribution[];
  };
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
  return useSWR<CockpitData>(getApiUrl(path), jsonFetcher, {
    refreshInterval: 5000,
    revalidateOnFocus: true,
  });
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
