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
  const { searchParams } = new URL(req.url);
  const limit = searchParams.get("limit") ?? "50";
  const reasonCode = searchParams.get("reason_code") ?? "";

  const params = new URLSearchParams({ limit });
  if (reasonCode) params.set("reason_code", reasonCode);

  const data = await safeFetch(`${getBackendApiBase()}/ops/order-rejections?${params}`, token);
  if (!data) {
    return NextResponse.json({
      available: false,
      note: "Backend unreachable",
      rejections: [],
    });
  }
  return NextResponse.json(data);
}
