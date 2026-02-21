import { NextRequest, NextResponse } from "next/server";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function getApiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

async function parseUpstreamBody(upstream: Response): Promise<unknown> {
  const text = await upstream.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

export async function proxyAuthorizedGet(request: NextRequest, upstreamPath: string): Promise<NextResponse> {
  const token = request.cookies.get("token")?.value;
  if (!token) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }

  try {
    const upstream = await fetch(`${getApiBase()}${upstreamPath}`, {
      method: "GET",
      cache: "no-store",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const body = await parseUpstreamBody(upstream);
    return NextResponse.json(body, { status: upstream.status });
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : "Portfolio proxy failed.";
    return NextResponse.json({ detail }, { status: 503 });
  }
}
