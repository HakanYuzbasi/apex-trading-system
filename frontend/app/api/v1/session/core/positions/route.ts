import { type NextRequest } from "next/server";
import { handleSessionPositions } from "@/app/api/v1/session/_lib/handlers";

export const dynamic = "force-dynamic";

export function GET(req: NextRequest) {
  return handleSessionPositions(req, "core");
}
