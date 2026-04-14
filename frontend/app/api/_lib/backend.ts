import { NextRequest } from "next/server";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export function getBackendApiBase(): string {
  const base = [
    process.env.APEX_API_URL,
    process.env.NEXT_PUBLIC_API_URL,
    DEFAULT_API_BASE,
  ].find((candidate) => candidate && candidate !== "undefined");

  return (base ?? DEFAULT_API_BASE).replace(/\/+$/, "");
}

export function getRequestToken(req: NextRequest): string | null {
  const authHeader = req.headers.get("authorization");
  if (authHeader?.toLowerCase().startsWith("bearer ")) {
    return authHeader.slice(7).trim();
  }

  return req.cookies.get("token")?.value ?? null;
}

export function buildAuthHeaders(token?: string | null): Record<string, string> {
  if (!token) {
    return {};
  }

  return { authorization: `Bearer ${token}` };
}
