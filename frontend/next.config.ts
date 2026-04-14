import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  // No rewrites — BFF route handlers in app/api/v1/ and app/api/auth/ proxy to
  // the backend using APEX_API_URL (http://apex-api:8000) which is reachable
  // from within the Docker network. A rewrite to localhost:8000 would be
  // unreachable from the Next.js container and would also bypass the BFF handlers.
};

export default nextConfig;
