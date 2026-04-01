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
  const data = await safeFetch("/ops/stress-state", token);

  if (!data || !data.state) {
    return NextResponse.json({
      state: null,
      note: data?.note ?? "backend unavailable",
    });
  }

  return NextResponse.json({
    state: data.state,
    note: data.note ?? null,
  });
}
