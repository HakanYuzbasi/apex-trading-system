import { NextRequest, NextResponse } from "next/server";
import { buildAuthHeaders, getBackendApiBase, getRequestToken } from "@/app/api/_lib/backend";
import {
  type SessionType,
  sanitizeSessionMetricsPayload,
  sanitizeSessionPositionsPayload,
  sanitizeSessionStatusPayload,
} from "@/app/api/v1/session/_lib/sessionPayloads";

export const dynamic = "force-dynamic";

export async function handleSessionMetrics(req: NextRequest, sessionType: SessionType) {
  const token = getRequestToken(req);
  try {
    const res = await fetch(
      `${getBackendApiBase()}/api/v1/session/${sessionType}/metrics`,
      { next: { revalidate: 0 }, headers: buildAuthHeaders(token) }
    );
    const data = await res.json().catch(() => null);
    const error = res.ok ? null : `upstream_${res.status}`;
    return NextResponse.json(sanitizeSessionMetricsPayload(sessionType, data, error, res.status));
  } catch (error) {
    return NextResponse.json(
      sanitizeSessionMetricsPayload(sessionType, null, error instanceof Error ? error.message : "upstream_unreachable", null)
    );
  }
}

export async function handleSessionPositions(req: NextRequest, sessionType: SessionType) {
  const token = getRequestToken(req);
  try {
    const res = await fetch(
      `${getBackendApiBase()}/api/v1/session/${sessionType}/positions`,
      { next: { revalidate: 0 }, headers: buildAuthHeaders(token) }
    );
    const data = await res.json().catch(() => null);
    const error = res.ok ? null : `upstream_${res.status}`;
    return NextResponse.json(sanitizeSessionPositionsPayload(sessionType, data, error, res.status));
  } catch (error) {
    return NextResponse.json(
      sanitizeSessionPositionsPayload(sessionType, null, error instanceof Error ? error.message : "upstream_unreachable", null)
    );
  }
}

export async function handleSessionStatus(req: NextRequest, sessionType: SessionType) {
  const token = getRequestToken(req);
  try {
    const res = await fetch(
      `${getBackendApiBase()}/api/v1/session/${sessionType}/status`,
      { next: { revalidate: 0 }, headers: buildAuthHeaders(token) }
    );
    const data = await res.json().catch(() => null);
    const error = res.ok ? null : `upstream_${res.status}`;
    return NextResponse.json(sanitizeSessionStatusPayload(sessionType, data, error, res.status));
  } catch (error) {
    return NextResponse.json(
      sanitizeSessionStatusPayload(sessionType, null, error instanceof Error ? error.message : "upstream_unreachable", null)
    );
  }
}
