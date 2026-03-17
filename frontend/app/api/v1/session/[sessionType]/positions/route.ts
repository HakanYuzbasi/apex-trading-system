import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const API_BASE = process.env.APEX_API_BASE ?? "http://127.0.0.1:8000";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ sessionType: string }> }
) {
  const { sessionType } = await params;
  if (!["core", "crypto", "unified"].includes(sessionType)) {
    return NextResponse.json({ error: "Invalid session type" }, { status: 400 });
  }

  try {
    const res = await fetch(
      `${API_BASE}/api/v1/session/${sessionType}/positions`,
      { next: { revalidate: 0 } }
    );
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      { session_type: sessionType, positions: [] },
      { status: 503 }
    );
  }
}
