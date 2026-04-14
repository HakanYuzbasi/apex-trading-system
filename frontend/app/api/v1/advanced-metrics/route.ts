import { NextRequest, NextResponse } from "next/server";
import { getBackendApiBase, getRequestToken, buildAuthHeaders } from "@/app/api/_lib/backend";

async function safeFetch(url: string, headers: Record<string, string>) {
  try {
    const res = await fetch(url, {
      headers,
      cache: "no-store",
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function GET(req: NextRequest) {
  const token = getRequestToken(req);
  const data = await safeFetch(
    `${getBackendApiBase()}/ops/advanced-metrics`,
    buildAuthHeaders(token)
  );
  if (!data) {
    return NextResponse.json({
      available: false,
      note: "Backend unreachable",
    });
  }
  return NextResponse.json(data);
}
