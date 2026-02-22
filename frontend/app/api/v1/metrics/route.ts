import { NextRequest, NextResponse } from "next/server";
import { sanitizeCount, sanitizeExecutionMetrics, sanitizeMoney } from "@/lib/metricGuards";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

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
    const combinedCapital = portfolioBalanceResp.ok
      ? sanitizeMoney(balanceData.total_equity, sanitized.capital)
      : sanitized.capital;
    const combinedOpenPositions = portfolioPositionsResp.ok
      ? Math.max(sanitized.open_positions, sanitizeCount(portfolioPositions.length, 0))
      : sanitized.open_positions;
    return NextResponse.json({
      status: String(data.status || "").toLowerCase() === "online",
      timestamp: (data.timestamp as string | null) ?? null,
      capital: combinedCapital,
      starting_capital: sanitized.starting_capital,
      daily_pnl: sanitized.daily_pnl,
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
