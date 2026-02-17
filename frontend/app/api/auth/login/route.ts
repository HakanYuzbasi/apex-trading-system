import { NextRequest, NextResponse } from "next/server";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function normalizeBase(base: string): string {
  return base.replace(/\/+$/, "");
}

function getApiBaseCandidates(): string[] {
  const configured = process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || "";
  const candidates = [
    configured,
    DEFAULT_API_BASE,
    "http://localhost:8000",
  ]
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .map(normalizeBase);

  return Array.from(new Set(candidates));
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON payload." }, { status: 400 });
  }

  const attempts: string[] = [];
  for (const base of getApiBaseCandidates()) {
    try {
      const upstream = await fetch(`${base}/auth/login`, {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload ?? {}),
      });
      const body = await upstream.json().catch(() => ({}));
      return NextResponse.json(body, { status: upstream.status });
    } catch (error: unknown) {
      const detail = error instanceof Error ? error.message : "unknown";
      attempts.push(`${base}: ${detail}`);
    }
  }

  return NextResponse.json(
    {
      detail: "Login proxy could not reach backend auth service.",
      attempts,
    },
    { status: 503 },
  );
}
