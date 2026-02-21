import { NextResponse } from "next/server";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function getApiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

export async function GET(): Promise<NextResponse> {
  try {
    const upstream = await fetch(`${getApiBase()}/public/cockpit`, {
      method: "GET",
      cache: "no-store",
    });

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => "Failed to load public cockpit.");
      return NextResponse.json({ detail }, { status: upstream.status });
    }

    const data = (await upstream.json()) as Record<string, unknown>;
    return NextResponse.json(data);
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : "Public cockpit proxy failed.";
    return NextResponse.json({ detail }, { status: 503 });
  }
}
