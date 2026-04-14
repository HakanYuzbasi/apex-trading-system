import { NextRequest, NextResponse } from "next/server";
import { buildAuthHeaders, getBackendApiBase, getRequestToken } from "@/app/api/_lib/backend";

export async function GET(req: NextRequest) {
  try {
    const token = getRequestToken(req);
    const res = await fetch(`${getBackendApiBase()}/ops/broker-mode/status`, {
      headers: buildAuthHeaders(token),
      cache: "no-store",
    });
    if (!res.ok) {
      return NextResponse.json({ broker_mode: "both" }, { status: res.status });
    }
    return NextResponse.json(await res.json());
  } catch {
    return NextResponse.json({ broker_mode: "both" });
  }
}

export async function POST(req: NextRequest) {
  try {
    const token = getRequestToken(req);
    const body = await req.json() as { target_mode?: string };
    const res = await fetch(`${getBackendApiBase()}/ops/broker-mode/change`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...buildAuthHeaders(token),
      },
      body: JSON.stringify({ target_mode: body.target_mode }),
      cache: "no-store",
    });
    const data = await res.json().catch(() => ({}));
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      { error: "Backend unreachable" },
      { status: 502 },
    );
  }
}
