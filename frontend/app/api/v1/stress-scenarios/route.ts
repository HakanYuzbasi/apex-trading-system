import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.APEX_API_URL ?? "http://localhost:8000";

async function safeFetch(url: string, token?: string) {
  try {
    const res = await fetch(url, {
      headers: token ? { authorization: `Bearer ${token}` } : {},
      cache: "no-store",
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function GET(req: NextRequest) {
  const token = req.headers.get("authorization")?.replace("Bearer ", "") ?? "";
  const data = await safeFetch(`${BACKEND}/ops/stress-scenarios`, token);
  if (!data) {
    return NextResponse.json({ available: false, note: "Backend unreachable" });
  }
  return NextResponse.json({ available: true, ...data });
}
