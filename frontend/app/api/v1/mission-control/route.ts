import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const BASE = process.env.APEX_API_BASE ?? "http://127.0.0.1:8000";

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
  const [state, regime, rlReport, univScores] = await Promise.all([
    safeFetch("/state", token),
    safeFetch("/ops/regime-transition", token),
    safeFetch("/ops/rl-governor", token),
    safeFetch("/ops/universe-scores", token),
  ]);

  // Extract the most critical fields
  const positions: Record<string, number> = state?.positions ?? {};
  const equity: number = state?.equity ?? 0;
  const dailyPnl: number = state?.daily_pnl ?? 0;
  const dailyPnlPct: number = equity > 0 ? (dailyPnl / equity) * 100 : 0;
  const maxPositions: number = state?.max_positions ?? 40;
  const positionCount: number = Object.keys(positions).length;
  const regime_str: string = state?.regime ?? "unknown";
  const vix: number = state?.vix ?? 0;
  const killSwitchActive: boolean = state?.kill_switch_active ?? false;
  const governorTier: string = state?.governor_tier ?? "GREEN";

  // Build top positions list (by |pnl_pct|)
  const positionDetails = (state?.position_details ?? []) as {
    symbol: string; pnl_pct: number; pnl: number; qty: number; side: string; signal_direction: string;
  }[];
  const topPositions = [...positionDetails]
    .sort((a, b) => Math.abs(b.pnl_pct) - Math.abs(a.pnl_pct))
    .slice(0, 5);

  return NextResponse.json({
    system: {
      regime: regime_str,
      vix,
      kill_switch_active: killSwitchActive,
      governor_tier: governorTier,
      equity: Math.round(equity),
    },
    risk_budget: {
      daily_pnl: Math.round(dailyPnl * 100) / 100,
      daily_pnl_pct: Math.round(dailyPnlPct * 100) / 100,
      position_count: positionCount,
      max_positions: maxPositions,
      positions_pct: maxPositions > 0 ? Math.round((positionCount / maxPositions) * 100) : 0,
    },
    top_positions: topPositions,
    predictive: {
      transition_probability: regime?.prediction?.probability ?? null,
      transition_direction: regime?.prediction?.direction ?? null,
      transition_size_mult: regime?.prediction?.size_multiplier ?? null,
      rl_epsilon: rlReport?.epsilon ?? null,
      rl_total_updates: rlReport?.total_updates ?? null,
      universe_scored: univScores?.report?.scored_count ?? null,
    },
    timestamp: new Date().toISOString(),
  });
}
