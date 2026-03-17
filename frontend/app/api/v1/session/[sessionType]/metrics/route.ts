import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const API_BASE = process.env.APEX_API_BASE ?? "http://127.0.0.1:8000";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ sessionType: string }> }
) {
  const { sessionType } = await params;
  if (!["core", "crypto", "unified"].includes(sessionType)) {
    return NextResponse.json({ error: "Invalid session type" }, { status: 400 });
  }

  const token = req.cookies.get("token")?.value;
  if (!token) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }

  try {
    const res = await fetch(
      `${API_BASE}/api/v1/session/${sessionType}/metrics`,
      {
        next: { revalidate: 0 },
        headers: { Authorization: `Bearer ${token}` },
      }
    );
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      { session_type: sessionType, capital: 0, sharpe_ratio: 0, sharpe_target: 1.5 },
      { status: 503 }
    );
  }
}
