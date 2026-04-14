import { NextRequest, NextResponse } from "next/server";
import { buildAuthHeaders, getBackendApiBase, getRequestToken } from "@/app/api/_lib/backend";

export const dynamic = "force-dynamic";
export const revalidate = 0;

function unavailablePayload(error: string | null = null) {
  return {
    available: false,
    error,
    timestamp: null,
    source: "backend_unavailable",
    equity: null,
    realized_pnl_today: null,
    active_margin: null,
    active_margin_utilization: null,
    sharpe_ratio: null,
    max_drawdown: null,
    curve_points: 0,
    sample_interval_seconds: null,
  };
}

export async function GET(req: NextRequest) {
  const token = getRequestToken(req);

  try {
    const res = await fetch(`${getBackendApiBase()}/ops/pitch-metrics`, {
      cache: "no-store",
      headers: buildAuthHeaders(token),
    });
    const data = await res.json().catch(() => null);
    if (!res.ok || !data || typeof data !== "object") {
      return NextResponse.json(unavailablePayload(`upstream_${res.status}`));
    }
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      unavailablePayload(error instanceof Error ? error.message : "upstream_unreachable"),
    );
  }
}
