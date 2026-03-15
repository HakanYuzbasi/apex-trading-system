import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const API_BASE = process.env.APEX_API_BASE ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/sessions`, {
      next: { revalidate: 0 },
    });
    if (!res.ok) {
      return NextResponse.json(
        { session_mode: "dual", sessions: [] },
        { status: res.status }
      );
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    // Fallback when backend is unreachable
    return NextResponse.json({
      session_mode: "dual",
      sessions: [
        { id: "core", label: "Core Strategy", enabled: true, description: "Equities, indices, and forex" },
        { id: "crypto", label: "Crypto Sleeve", enabled: true, description: "Cryptocurrency trading" },
      ],
    });
  }
}
