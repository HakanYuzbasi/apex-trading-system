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
  const data = await safeFetch(`${BACKEND}/ops/cross-asset-pairs`, token);
  if (!data) {
    return NextResponse.json({
      available: false,
      note: "Backend unreachable",
      n_pairs: 0,
      active_pairs: [],
    });
  }
  return NextResponse.json(data);
}
