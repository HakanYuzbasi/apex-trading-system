import { NextResponse } from "next/server";
import { sanitizeExecutionMetrics } from "@/lib/metricGuards";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function getApiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

export async function GET(): Promise<NextResponse> {
  try {
    const upstream = await fetch(`${getApiBase()}/public/metrics`, {
      method: "GET",
      cache: "no-store",
    });

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => "Failed to load public metrics.");
      return NextResponse.json({ detail }, { status: upstream.status });
    }

    const data = (await upstream.json()) as Record<string, unknown>;
    const sanitized = sanitizeExecutionMetrics(data);
    return NextResponse.json({
      status: String(data.status || "").toLowerCase() === "online",
      timestamp: (data.timestamp as string | null) ?? null,
      capital: sanitized.capital,
      starting_capital: sanitized.starting_capital,
      daily_pnl: sanitized.daily_pnl,
      total_pnl: sanitized.total_pnl,
      max_drawdown: sanitized.max_drawdown,
      sharpe_ratio: sanitized.sharpe_ratio,
      win_rate: sanitized.win_rate,
      open_positions: sanitized.open_positions,
      option_positions: sanitized.option_positions,
      open_positions_total: sanitized.open_positions_total,
      trades_count: sanitized.trades_count,
    });
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : "Public metrics proxy failed.";
    return NextResponse.json({ detail }, { status: 503 });
  }
}
