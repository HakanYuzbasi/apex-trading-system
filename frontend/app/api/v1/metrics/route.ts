import { NextRequest, NextResponse } from "next/server";
import { sanitizeCount, sanitizeExecutionMetrics, sanitizeMoney } from "@/lib/metricGuards";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";
export const dynamic = "force-dynamic";
export const revalidate = 0;

type MetricsSnapshotCache = {
  updated_at_ms: number;
  capital: number;
  open_positions: number;
};

let LAST_GOOD_METRICS_SNAPSHOT: MetricsSnapshotCache | null = null;
const METRICS_SNAPSHOT_CACHE_TTL_MS = Number(process.env.APEX_METRICS_UI_STABILITY_MS || "30000");

function getApiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  const token = request.cookies.get("token")?.value;
  if (!token) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }

  try {
    const authHeaders = {
      Authorization: `Bearer ${token}`,
    };
    const [statusResp, portfolioBalanceResp, portfolioPositionsResp] = await Promise.all([
      fetch(`${getApiBase()}/status`, {
        method: "GET",
        cache: "no-store",
        headers: authHeaders,
      }),
      fetch(`${getApiBase()}/portfolio/balance`, {
        method: "GET",
        cache: "no-store",
        headers: authHeaders,
      }),
      fetch(`${getApiBase()}/portfolio/positions`, {
        method: "GET",
        cache: "no-store",
        headers: authHeaders,
      }),
    ]);

    if (!statusResp.ok) {
      const detail = await statusResp.text().catch(() => "Failed to load status.");
      return NextResponse.json({ detail }, { status: statusResp.status });
    }

    const data = (await statusResp.json()) as Record<string, unknown>;
    const balanceData = portfolioBalanceResp.ok
      ? ((await portfolioBalanceResp.json().catch(() => ({}))) as Record<string, unknown>)
      : {};
    const portfolioPositions = portfolioPositionsResp.ok
      ? ((await portfolioPositionsResp.json().catch(() => [])) as unknown[])
      : [];
    const sanitized = sanitizeExecutionMetrics(data);
    const nowMs = Date.now();
    
    const combinedCapital = portfolioBalanceResp.ok ? sanitizeMoney(balanceData.total_equity) : sanitized.capital;
    const combinedOpenPositions = sanitized.open_positions;

    return NextResponse.json({
      status: String(data.status || "").toLowerCase() === "online",
      timestamp: (data.timestamp as string | null) ?? null,
      capital: combinedCapital,
      starting_capital: sanitized.starting_capital,
      daily_pnl: sanitized.daily_pnl,
      daily_pnl_realized: sanitizeMoney(data.daily_pnl_realized, sanitized.daily_pnl),
      daily_pnl_source: String(data.daily_pnl_source || "inferred"),
      total_pnl: sanitized.total_pnl,
      max_drawdown: sanitized.max_drawdown,
      sharpe_ratio: sanitized.sharpe_ratio,
      win_rate: sanitized.win_rate,
      open_positions: combinedOpenPositions,
      trades_count: sanitized.trades_count,
    });
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : "Metrics proxy failed.";
    return NextResponse.json({ detail }, { status: 503 });
  }
}
