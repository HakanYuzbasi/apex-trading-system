import { NextResponse } from "next/server";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function getApiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

export async function GET(): Promise<NextResponse> {
  try {
    const upstream = await fetch(`${getApiBase()}/auth/plans`, {
      method: "GET",
      cache: "no-store",
    });
    const payload = await upstream.json().catch(() => null);
    if (!upstream.ok) {
      return NextResponse.json(
        { detail: "Failed to load plan catalog." },
        { status: upstream.status || 503 },
      );
    }
    return NextResponse.json(Array.isArray(payload) ? payload : []);
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : "Plan catalog proxy failed.";
    return NextResponse.json({ detail }, { status: 503 });
  }
}

