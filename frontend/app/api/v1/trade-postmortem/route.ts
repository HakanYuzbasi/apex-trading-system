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
  const n = parseInt(url.searchParams.get("n") ?? "20", 10);

  const data = await safeFetch(`/ops/trade-postmortem?n=${n}`, token);

  if (!data) {
    return NextResponse.json({
      recent: [],
      summary: { total: 0, win_rate: 0, avg_pnl_pct: 0, verdict_counts: {}, failure_counts: {} },
      note: "backend unavailable",
    });
  }

  return NextResponse.json({
    recent: data.recent ?? [],
    summary: data.summary ?? {},
    note: data.note ?? null,
  });
}
