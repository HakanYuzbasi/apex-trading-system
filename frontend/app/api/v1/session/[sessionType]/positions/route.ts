import { NextRequest, NextResponse } from "next/server";
import { buildAuthHeaders, getBackendApiBase, getRequestToken } from "@/app/api/_lib/backend";
import {
  isValidSessionType,
  sanitizeSessionPositionsPayload,
} from "@/app/api/v1/session/_lib/sessionPayloads";

export const dynamic = "force-dynamic";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ sessionType: string }> }
) {
  const { sessionType } = await params;
  if (!isValidSessionType(sessionType)) {
    return NextResponse.json({ error: "Invalid session type" }, { status: 400 });
  }

  const token = getRequestToken(req);

  try {
    const res = await fetch(
      `${getBackendApiBase()}/api/v1/session/${sessionType}/positions`,
      {
        next: { revalidate: 0 },
        headers: buildAuthHeaders(token),
      }
    );
    const data = await res.json().catch(() => null);
    const error = res.ok ? null : `upstream_${res.status}`;
    return NextResponse.json(
      sanitizeSessionPositionsPayload(sessionType, data, error, res.status),
    );
  } catch (error) {
    return NextResponse.json(
      sanitizeSessionPositionsPayload(
        sessionType,
        null,
        error instanceof Error ? error.message : "upstream_unreachable",
        null,
      ),
    );
  }
}
