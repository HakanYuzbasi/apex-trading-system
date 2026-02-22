import { NextRequest, NextResponse } from "next/server";
import { proxyAuthorizedGet } from "@/app/api/v1/portfolio/proxy";

export async function GET(request: NextRequest): Promise<NextResponse> {
  return proxyAuthorizedGet(request, "/portfolio/positions");
}
