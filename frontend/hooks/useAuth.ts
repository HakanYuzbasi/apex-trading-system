"use client";

import { useCallback, useEffect, useState } from "react";

const STORAGE_ACCESS = "apex_access_token";
const STORAGE_REFRESH = "apex_refresh_token";
const STORAGE_USER = "apex_user";

export type SubscriptionTier = "free" | "basic" | "pro" | "enterprise";

export interface AuthUser {
  user_id: string;
  username: string;
  email?: string | null;
  roles: string[];
  tier: SubscriptionTier;
}

export interface AuthState {
  user: AuthUser | null;
  accessToken: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(STORAGE_ACCESS);
}

function getStoredRefresh(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(STORAGE_REFRESH);
}

export function getStoredUser(): AuthUser | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(STORAGE_USER);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  } catch {
    return null;
  }
}

function setStored(token: string | null, refresh: string | null, user: AuthUser | null): void {
  if (typeof window === "undefined") return;
  if (token) localStorage.setItem(STORAGE_ACCESS, token);
  else localStorage.removeItem(STORAGE_ACCESS);
  if (refresh) localStorage.setItem(STORAGE_REFRESH, refresh);
  else localStorage.removeItem(STORAGE_REFRESH);
  if (user) localStorage.setItem(STORAGE_USER, JSON.stringify(user));
  else localStorage.removeItem(STORAGE_USER);

  // Keep server-side middleware/auth routes aligned with client auth state.
  if (typeof document !== "undefined") {
    const secure = window.location.protocol === "https:" ? "; secure" : "";
    if (token) {
      document.cookie = `token=${encodeURIComponent(token)}; path=/; max-age=1800; samesite=lax${secure}`;
    } else {
      document.cookie = `token=; path=/; max-age=0; samesite=lax${secure}`;
    }
  }
}

export function useAuth() {
  const [state, setState] = useState<AuthState>({
    user: null,
    accessToken: null,
    isLoading: true,
    isAuthenticated: false,
  });

  const setUser = useCallback((user: AuthUser | null, accessToken: string | null, refreshToken: string | null) => {
    setStored(accessToken, refreshToken, user);
    setState({
      user,
      accessToken,
      isLoading: false,
      isAuthenticated: !!user && !!accessToken,
    });
  }, []);

  const login = useCallback(async (username: string, password: string): Promise<{ ok: boolean; error?: string }> => {
    try {
      const res = await fetch(`/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        return { ok: false, error: (data as { detail?: string }).detail || "Login failed" };
      }
      const access = (data as { access_token: string }).access_token;
      const refresh = (data as { refresh_token: string }).refresh_token;
      const meRes = await fetch(`/api/auth/me`, {
        headers: { Authorization: `Bearer ${access}` },
      });
      const me = meRes.ok ? (await meRes.json()) as AuthUser : null;
      const user: AuthUser = me || {
        user_id: "",
        username,
        email: null,
        roles: ["user"],
        tier: "free",
      };
      if (me) {
        user.user_id = me.user_id;
        user.email = me.email;
        user.roles = me.roles ?? ["user"];
        user.tier = (me.tier as SubscriptionTier) ?? "free";
      }
      setUser(user, access, refresh);
      return { ok: true };
    } catch (e) {
      const message = e instanceof Error ? e.message : "Login failed";
      return { ok: false, error: message };
    }
  }, [setUser]);

  const register = useCallback(async (username: string, email: string, password: string): Promise<{ ok: boolean; error?: string }> => {
    try {
      const res = await fetch(`/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        return { ok: false, error: (data as { detail?: string }).detail || "Registration failed" };
      }
      const access = (data as { access_token: string }).access_token;
      const refresh = (data as { refresh_token: string }).refresh_token;
      const user: AuthUser = {
        user_id: "",
        username,
        email,
        roles: ["user"],
        tier: "free",
      };
      const meRes = await fetch(`/api/auth/me`, { headers: { Authorization: `Bearer ${access}` } });
      if (meRes.ok) {
        const me = (await meRes.json()) as AuthUser;
        user.user_id = me.user_id;
        user.roles = me.roles ?? ["user"];
        user.tier = (me.tier as SubscriptionTier) ?? "free";
      }
      setUser(user, access, refresh);
      return { ok: true };
    } catch (e) {
      const message = e instanceof Error ? e.message : "Registration failed";
      return { ok: false, error: message };
    }
  }, [setUser]);

  const logout = useCallback(() => {
    setUser(null, null, null);
  }, [setUser]);

  const refreshUser = useCallback(async () => {
    const token = getStoredToken();
    if (!token) {
      setState((s) => ({ ...s, isLoading: false, isAuthenticated: false }));
      return;
    }
    try {
      const res = await fetch(`/api/auth/me`, { headers: { Authorization: `Bearer ${token}` } });
      if (res.ok) {
        const user = (await res.json()) as AuthUser;
        setState((s) => ({
          ...s,
          user,
          accessToken: token,
          isLoading: false,
          isAuthenticated: true,
        }));
        localStorage.setItem(STORAGE_USER, JSON.stringify(user));
      } else {
        const refresh = getStoredRefresh();
        if (refresh) {
          const rRes = await fetch(`/api/auth/refresh`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ refresh_token: refresh }),
          });
          const rData = rRes.ok ? (await rRes.json()) as { access_token: string; refresh_token: string } : null;
          if (rData) {
            const meRes = await fetch(`/api/auth/me`, {
              headers: { Authorization: `Bearer ${rData.access_token}` },
            });
            const user = meRes.ok ? ((await meRes.json()) as AuthUser) : null;
            setUser(user || state.user, rData.access_token, rData.refresh_token);
            return;
          }
        }
        setUser(null, null, null);
      }
    } catch {
      setState((s) => ({ ...s, isLoading: false, isAuthenticated: false }));
    }
  }, [setUser, state.user]);

  useEffect(() => {
    const token = getStoredToken();
    const cachedUser = getStoredUser();
    if (token && cachedUser) {
      setState({
        user: cachedUser,
        accessToken: token,
        isLoading: false,
        isAuthenticated: true,
      });
      refreshUser();
    } else {
      setState((s) => ({ ...s, isLoading: false }));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    ...state,
    login,
    register,
    logout,
    refreshUser,
    getToken: getStoredToken,
  };
}
