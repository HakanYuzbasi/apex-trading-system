"use client";

import React, { createContext, useContext, useMemo } from "react";
import { useAuth, type AuthUser } from "@/hooks/useAuth";

type AuthContextValue = ReturnType<typeof useAuth>;

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const auth = useAuth();
  const value = useMemo(() => auth, [auth.user, auth.accessToken, auth.isLoading, auth.isAuthenticated]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuthContext(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuthContext must be used within AuthProvider");
  return ctx;
}
