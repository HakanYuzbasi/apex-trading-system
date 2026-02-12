"use client";

import { useCallback } from "react";
import { apiFetch, apiJson, getApiUrl } from "@/lib/api";
import { useAuth } from "./useAuth";

/**
 * Authenticated API calls using the current user's token.
 * Use for SaaS feature endpoints that require auth.
 */
export function useApi() {
  const { accessToken, isAuthenticated } = useAuth();

  const authFetch = useCallback(
    async (path: string, options: RequestInit = {}): Promise<Response> => {
      return apiFetch(path, { ...options, token: accessToken });
    },
    [accessToken]
  );

  const authJson = useCallback(
    async <T>(path: string, options: RequestInit = {}): Promise<T> => {
      return apiJson<T>(path, { ...options, token: accessToken });
    },
    [accessToken]
  );

  return {
    getApiUrl,
    authFetch,
    authJson,
    isAuthenticated,
    token: accessToken,
  };
}
