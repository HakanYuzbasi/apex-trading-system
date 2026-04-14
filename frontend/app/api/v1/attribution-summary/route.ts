import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const BASE = process.env.APEX_API_URL ?? process.env.APEX_API_BASE ?? "http://127.0.0.1:8000";

async function safeFetch(path: string, token?: string) {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await fetch(`${BASE}${path}`, { headers, cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function GET(req: Request) {
  const token = req.headers.get("authorization")?.replace("Bearer ", "");
  const url = new URL(req.url);
  const lookback = url.searchParams.get("lookback_days") ?? "30";

  const data = await safeFetch(`/ops/attribution-summary?lookback_days=${lookback}`, token);

  if (!data) {
    return NextResponse.json({
      summary: {},
      signal_sources: { lookback_days: Number(lookback), by_signal_source: {} },
      note: "backend unavailable",
    });
  }

  return NextResponse.json({
    summary: data.summary ?? {},
    signal_sources: data.signal_sources ?? { lookback_days: Number(lookback), by_signal_source: {} },
    note: data.note ?? null,
  });
}
