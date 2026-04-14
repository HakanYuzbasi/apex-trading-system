import { NextRequest, NextResponse } from "next/server";
import { getBackendApiBase } from "@/app/api/_lib/backend";

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
  const url = new URL(req.url);
  const n = url.searchParams.get("n") ?? "50";
  const data = await safeFetch(`${getBackendApiBase()}/ops/alerts?n=${n}`, token);
  if (!data) {
    return NextResponse.json({
      available: false,
      note: "Backend unreachable",
      alerts: [],
    });
  }
  return NextResponse.json(data);
}
