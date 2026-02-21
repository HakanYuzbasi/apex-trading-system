import { NextRequest, NextResponse } from "next/server";
import { proxyGetAuth } from "../proxy";

export async function GET(request: NextRequest): Promise<NextResponse> {
  return proxyGetAuth(request, "/auth/me");
}
