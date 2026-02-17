import { NextRequest, NextResponse } from "next/server";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function getApiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  const token = request.cookies.get("token")?.value;
  if (!token) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }
  const params = request.nextUrl.searchParams;
  const month = params.get("month");
  const lookback = params.get("lookback") || "1000";
  const query = new URLSearchParams();
  query.set("lookback", lookback);
  if (month) query.set("month", month);
  try {
    const upstream = await fetch(`${getApiBase()}/api/v1/mandate-copilot/reports/monthly?${query.toString()}`, {
      method: "GET",
      cache: "no-store",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const body = await upstream.json().catch(() => ({}));
    return NextResponse.json(body, { status: upstream.status });
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : "Monthly model risk report proxy failed.";
    return NextResponse.json({ detail }, { status: 503 });
  }
}

