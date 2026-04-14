import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function getApiBase(): string {
  return (process.env.APEX_API_URL || process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  const token = request.cookies.get("token")?.value;
  if (!token) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }

  const params = request.nextUrl.searchParams;
  const symbol = String(params.get("symbol") ?? "").trim();
  const planId = String(params.get("plan_id") ?? "").trim();
  if (!symbol && !planId) {
    return NextResponse.json({ detail: "symbol or plan_id is required" }, { status: 400 });
  }

  const limit = params.get("limit") ?? "250";
  const days = params.get("days") ?? "7";
  const includeRaw = params.get("include_raw") ?? "true";
  const query = new URLSearchParams({
    limit,
    days,
    include_raw: includeRaw,
  });

  try {
    const upstreamPath = planId
      ? `/api/v1/replay-inspector/plan/${encodeURIComponent(planId)}`
      : `/api/v1/replay-inspector/${encodeURIComponent(symbol)}`;
    const upstream = await fetch(
      `${getApiBase()}${upstreamPath}?${query.toString()}`,
      {
        method: "GET",
        cache: "no-store",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      },
    );

    const data = await upstream.json().catch(() => ({ detail: "Replay inspector upstream failed." }));
    return NextResponse.json(data, { status: upstream.status });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Replay inspector proxy failed.";
    return NextResponse.json({ detail }, { status: 502 });
  }
}
