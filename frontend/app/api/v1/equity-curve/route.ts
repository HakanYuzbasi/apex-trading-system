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
  const url = new URL(req.url);
  const points = url.searchParams.get("points") ?? "200";

  const data = await safeFetch(`/ops/equity-curve?points=${points}`, token);

  if (!data) {
    return NextResponse.json({
      curve: [],
      drawdown: [],
      peak: 0,
      current: 0,
      drawdown_pct: 0,
      total_points: 0,
      note: "backend unavailable",
    });
  }

  return NextResponse.json({
    curve: data.curve ?? [],
    drawdown: data.drawdown ?? [],
    peak: data.peak ?? 0,
    current: data.current ?? 0,
    drawdown_pct: data.drawdown_pct ?? 0,
    total_points: data.total_points ?? 0,
    note: data.note ?? null,
  });
}
