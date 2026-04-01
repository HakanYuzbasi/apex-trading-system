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
  const data = await safeFetch(`${BACKEND}/ops/ab-gate`, token);
  if (!data) {
    return NextResponse.json({ available: false, note: "Backend unreachable" });
  }
  return NextResponse.json(data);
}

export async function POST(req: NextRequest) {
  const token = req.headers.get("authorization")?.replace("Bearer ", "") ?? "";
  try {
    const body = await req.json();
    const res = await fetch(`${BACKEND}/ops/ab-gate/register`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "unknown error" }));
      return NextResponse.json(err, { status: res.status });
    }
    return NextResponse.json(await res.json());
  } catch {
    return NextResponse.json({ error: "Failed to register challenger" }, { status: 500 });
  }
}
