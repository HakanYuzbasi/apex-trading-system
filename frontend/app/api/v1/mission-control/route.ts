import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const BASE = process.env.APEX_API_URL ?? process.env.APEX_API_BASE ?? "http://127.0.0.1:8000";

async function safeFetch(path: string, token?: string) {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await fetch(`${BASE}${path}`, { headers, cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function GET(req: Request) {
  const token = req.headers.get("authorization")?.replace("Bearer ", "");

  // Fetch all mission-critical data in parallel
  const [state, rlReport, cockpit] = await Promise.all([
    safeFetch("/state", token),
    safeFetch("/ops/rl-governor", token),
    safeFetch("/api/v1/cockpit", token),
  ]);

  // --- Equity & Capital ---
  const equity: number = state?.capital ?? state?.equity ?? 0;

  // --- Daily P&L ---
  const dailyPnl: number = state?.daily_pnl ?? 0;
  const dailyPnlPct: number = equity > 0 ? (dailyPnl / equity) * 100 : 0;

  // --- Regime from state (engine writes vix/regime when available) ---
  const regime_str: string = state?.regime ?? state?.market_regime ?? "neutral";
  const vix: number = state?.vix ?? state?.vix_level ?? 0;
  const killSwitchActive: boolean = state?.kill_switch_active ?? false;

  // Governor tier derived from drawdown & meta confidence
  const drawdownPct = Math.abs(state?.max_drawdown ?? state?.drawdown ?? 0) * 100;
  const metaConf = state?.meta_confidence_score ?? 1.0;
  let governorTier = "GREEN";
  if (drawdownPct > 5 || metaConf < 0.6) governorTier = "RED";
  else if (drawdownPct > 2 || metaConf < 0.8) governorTier = "YELLOW";

  // --- Position count from state ---
  const openPositions: number = state?.open_positions ?? 0;
  const maxPositions: number = state?.max_positions ?? 40;

  // --- Top positions from cockpit (has enriched position list) ---
  type CockpitPosition = {
    symbol: string;
    pnl_pct: number;
    pnl: number;
    qty: number;
    side: string;
    signal_direction: string;
  };
  const cockpitPositions: CockpitPosition[] = (cockpit?.positions ?? []) as CockpitPosition[];
  const topPositions = [...cockpitPositions]
    .sort((a, b) => Math.abs(b.pnl_pct) - Math.abs(a.pnl_pct))
    .slice(0, 5)
    .map((p) => ({
      symbol: p.symbol,
      pnl_pct: p.pnl_pct ?? 0,
      pnl: p.pnl ?? 0,
      qty: p.qty ?? 0,
      side: p.side ?? "LONG",
      signal_direction: p.signal_direction ?? "UNKNOWN",
    }));

  // --- Predictive fields from rl-governor (regime-transition returns null in Docker mode) ---
  return NextResponse.json({
    system: {
      regime: regime_str,
      vix: Math.round(vix * 100) / 100,
      kill_switch_active: killSwitchActive,
      governor_tier: governorTier,
      equity: Math.round(equity),
      meta_confidence: Math.round(metaConf * 100) / 100,
      survival_probability: Math.round((state?.survival_probability ?? 1.0) * 100) / 100,
    },
    risk_budget: {
      daily_pnl: Math.round(dailyPnl * 100) / 100,
      daily_pnl_pct: Math.round(dailyPnlPct * 100) / 100,
      realized_pnl: Math.round((state?.realized_pnl ?? 0) * 100) / 100,
      unrealized_pnl: Math.round((state?.unrealized_pnl ?? 0) * 100) / 100,
      drawdown_pct: Math.round(drawdownPct * 100) / 100,
      active_margin: Math.round((state?.active_margin ?? 0) * 100) / 100,
      position_count: openPositions,
      max_positions: maxPositions,
      positions_pct: maxPositions > 0 ? Math.round((openPositions / maxPositions) * 100) : 0,
    },
    top_positions: topPositions,
    predictive: {
      transition_probability: null,
      transition_direction: null,
      transition_size_mult: null,
      rl_epsilon: rlReport?.epsilon ?? null,
      rl_total_updates: rlReport?.total_updates ?? null,
      universe_scored: null,
      bayesian_vol_prob: state?.bayesian_vol_prob ?? null,
    },
    timestamp: new Date().toISOString(),
  });
}
