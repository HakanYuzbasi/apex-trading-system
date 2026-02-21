import { NextRequest, NextResponse } from "next/server";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function normalizeBase(base: string): string {
  return base.replace(/\/+$/, "");
}

function getApiBaseCandidates(): string[] {
  const configured = process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || "";
  const candidates = [configured, DEFAULT_API_BASE, "http://localhost:8000"]
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .map(normalizeBase);

  return Array.from(new Set(candidates));
}

async function parseUpstreamBody(upstream: Response): Promise<unknown> {
  const text = await upstream.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

export async function proxyPostJson(request: NextRequest, upstreamPath: string): Promise<NextResponse> {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON payload." }, { status: 400 });
  }

  const attempts: string[] = [];
  for (const base of getApiBaseCandidates()) {
    try {
      const upstream = await fetch(`${base}${upstreamPath}`, {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload ?? {}),
      });
      const body = await parseUpstreamBody(upstream);
      if (upstream.status >= 500) {
        attempts.push(`${base}: status ${upstream.status}`);
        continue;
      }
      return NextResponse.json(body, { status: upstream.status });
    } catch (error: unknown) {
      const detail = error instanceof Error ? error.message : "unknown";
      attempts.push(`${base}: ${detail}`);
    }
  }

  return NextResponse.json(
    {
      detail: "Auth proxy could not reach a healthy backend auth service.",
      attempts,
    },
    { status: 503 },
  );
}

export async function proxyGetAuth(request: NextRequest, upstreamPath: string): Promise<NextResponse> {
  const auth = request.headers.get("authorization");
  if (!auth) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }

  const attempts: string[] = [];
  for (const base of getApiBaseCandidates()) {
    try {
      const upstream = await fetch(`${base}${upstreamPath}`, {
        method: "GET",
        cache: "no-store",
        headers: { Authorization: auth },
      });
      const body = await parseUpstreamBody(upstream);
      if (upstream.status >= 500) {
        attempts.push(`${base}: status ${upstream.status}`);
        continue;
      }
      return NextResponse.json(body, { status: upstream.status });
    } catch (error: unknown) {
      const detail = error instanceof Error ? error.message : "unknown";
      attempts.push(`${base}: ${detail}`);
    }
  }

  return NextResponse.json(
    {
      detail: "Auth proxy could not reach a healthy backend auth service.",
      attempts,
    },
    { status: 503 },
  );
}
