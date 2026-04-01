import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const revalidate = 0;

const BASE = process.env.APEX_API_BASE ?? "http://127.0.0.1:8000";

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
  const data = await safeFetch("/ops/missed-opportunities", token);

  if (!data) {
    return NextResponse.json({
      pending_count: 0,
      completed_count: 0,
      recent_pending: [],
      report: { total_missed: 0, total_missed_pnl_5d: 0, total_missed_pnl_10d: 0, by_filter_reason: {}, by_regime: {}, top_missed_symbols: [], generated_at: "" },
      note: "backend unavailable",
    });
  }

  return NextResponse.json({
    pending_count: data.pending_count ?? 0,
    completed_count: data.completed_count ?? 0,
    recent_pending: data.recent_pending ?? [],
    report: data.report ?? {},
    note: data.note ?? null,
  });
}
