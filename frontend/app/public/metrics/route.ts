import { NextResponse } from "next/server";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function getApiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

function asNumber(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
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
    return NextResponse.json({
      status: String(data.status || "").toLowerCase() === "online",
      timestamp: (data.timestamp as string | null) ?? null,
      capital: asNumber(data.capital),
      daily_pnl: asNumber(data.daily_pnl),
      total_pnl: asNumber(data.total_pnl),
      max_drawdown: asNumber(data.max_drawdown),
      sharpe_ratio: asNumber(data.sharpe_ratio),
      win_rate: asNumber(data.win_rate),
      open_positions: asNumber(data.open_positions),
      trades_count: asNumber(data.total_trades),
    });
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : "Public metrics proxy failed.";
    return NextResponse.json({ detail }, { status: 503 });
  }
}
