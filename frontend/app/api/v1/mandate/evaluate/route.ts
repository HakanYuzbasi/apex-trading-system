import { NextRequest, NextResponse } from "next/server";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function getApiBase(): string {
  return (process.env.NEXT_PUBLIC_API_URL || process.env.APEX_API_URL || DEFAULT_API_BASE).replace(/\/+$/, "");
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  const token = request.cookies.get("token")?.value;
  if (!token) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON payload." }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${getApiBase()}/api/v1/mandate-copilot/evaluate`, {
      method: "POST",
      cache: "no-store",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(payload ?? {}),
    });
    const body = await upstream.json().catch(() => ({}));
    return NextResponse.json(body, { status: upstream.status });
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : "Mandate copilot proxy failed.";
    return NextResponse.json({ detail }, { status: 503 });
  }
}

