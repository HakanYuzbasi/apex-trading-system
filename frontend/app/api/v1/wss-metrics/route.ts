import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export async function GET() {
  const apiBase = process.env.APEX_API_BASE ?? DEFAULT_API_BASE;
  try {
    const res = await fetch(`${apiBase}/ops/wss-metrics`, {
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });
    if (!res.ok)
      return NextResponse.json(
        { error: `Backend returned ${res.status}` },
        { status: res.status }
      );
    return NextResponse.json(await res.json());
  } catch {
    return NextResponse.json({ error: "WSS metrics unavailable" }, { status: 503 });
  }
}
