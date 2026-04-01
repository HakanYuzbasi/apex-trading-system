import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export async function GET(req: NextRequest) {
  const apiBase = process.env.APEX_API_BASE ?? DEFAULT_API_BASE;
  const { searchParams } = new URL(req.url);
  const lookback = searchParams.get("lookback_days") ?? "7";
  try {
    const res = await fetch(
      `${apiBase}/ops/factor-pnl?lookback_days=${lookback}`,
      {
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
      }
    );
    if (!res.ok) {
      return NextResponse.json(
        { error: `Backend returned ${res.status}` },
        { status: res.status }
      );
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Factor P&L service unavailable" },
      { status: 503 }
    );
  }
}
