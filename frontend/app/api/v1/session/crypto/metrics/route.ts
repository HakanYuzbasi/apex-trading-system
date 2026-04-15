import { type NextRequest } from "next/server";
import { handleSessionMetrics } from "@/app/api/v1/session/_lib/handlers";

export const dynamic = "force-dynamic";

export function GET(req: NextRequest) {
  return handleSessionMetrics(req, "crypto");
}
