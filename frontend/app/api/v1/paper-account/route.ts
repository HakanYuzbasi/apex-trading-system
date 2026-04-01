import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.APEX_API_URL ?? "http://localhost:8000";

async function safeFetch(url: string, token?: string) {
  try {
    const res = await fetch(url, {
      headers: token ? { authorization: `Bearer ${token}` } : {},
      cache: "no-store",
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function GET(req: NextRequest) {
  const token = req.headers.get("authorization")?.replace("Bearer ", "") ?? "";
  const data = await safeFetch(`${BACKEND}/ops/paper-account`, token);
  if (!data) {
    return NextResponse.json({
      available: false,
      note: "Backend unreachable",
      open_positions: 0,
      closed_trades: 0,
      paper_total_pnl: 0,
      live_total_pnl: 0,
      implementation_shortfall_usd: 0,
      shortfall_pct: 0,
      avg_shortfall_per_trade: 0,
      win_rates: { paper: 0, live: 0, n: 0 },
      recent_trades: [],
    });
  }
  return NextResponse.json(data);
}
