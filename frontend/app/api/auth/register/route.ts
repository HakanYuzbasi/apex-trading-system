import { NextRequest, NextResponse } from "next/server";
import { proxyPostJson } from "../proxy";

export async function POST(request: NextRequest): Promise<NextResponse> {
  return proxyPostJson(request, "/auth/register");
}
