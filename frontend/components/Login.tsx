"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { ArrowRight, Lock, Moon, Radar, ShieldCheck, Sun, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useTheme } from "@/components/theme/ThemeProvider";

const DEFAULT_SESSION_COOKIE_MAX_AGE_SECONDS = 24 * 60 * 60;

function decodeJwtPayload(token: string): Record<string, unknown> | null {
  const parts = token.split(".");
  if (parts.length < 2) return null;
  try {
    const payloadB64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    const padded = payloadB64 + "=".repeat((4 - (payloadB64.length % 4)) % 4);
    const decoded = atob(padded);
    return JSON.parse(decoded) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function computeSessionCookieMaxAgeSeconds(token: string, expiresIn?: unknown): number {
  if (typeof expiresIn === "number" && Number.isFinite(expiresIn) && expiresIn > 0) {
    return Math.floor(expiresIn);
  }
  const payload = decodeJwtPayload(token);
  const exp = payload?.exp;
  if (typeof exp === "number" && Number.isFinite(exp)) {
    return Math.max(0, Math.floor(exp - Date.now() / 1000));
  }
  return DEFAULT_SESSION_COOKIE_MAX_AGE_SECONDS;
}

export default function Login() {
  const { theme, toggleTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();
  const reason = searchParams.get("reason");

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) {
        const errorBody = await res.json().catch(() => ({})) as { detail?: string; attempts?: string[] };
        let message = typeof errorBody?.detail === "string" ? errorBody.detail : "Invalid credentials";
        if (res.status === 503) {
          const attempts = Array.isArray(errorBody.attempts) ? errorBody.attempts.join(" | ") : "";
          message = attempts ? `${message} (${attempts})` : message;
        }
        throw new Error(message);
      }

      const data = await res.json();
      const accessToken = String(data.access_token || "");
      const maxAge = computeSessionCookieMaxAgeSeconds(accessToken, data.expires_in);
      const secure = window.location.protocol === "https:" ? "; secure" : "";
      document.cookie = `token=${encodeURIComponent(accessToken)}; path=/; max-age=${maxAge}; samesite=lax${secure}`;
      router.push("/dashboard");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Login failed";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="apex-shell min-h-screen px-4 py-6 sm:px-6 lg:px-10">
      <div className="mx-auto grid w-full max-w-6xl gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <section className="apex-panel apex-fade-up hidden min-h-[560px] rounded-3xl p-8 lg:flex lg:flex-col lg:justify-between">
          <header className="space-y-4">
            <p className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold tracking-wide text-primary">
              <Radar className="h-3.5 w-3.5" />
              APEX Trading Command
            </p>
            <h1 className="max-w-xl text-4xl font-semibold leading-tight text-foreground">
              Institutional execution monitoring with hardened risk controls.
            </h1>
            <p className="max-w-lg text-base text-muted-foreground">
              Review live exposure, governor behavior, and reconciliation health from a single operational console.
            </p>
          </header>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-2xl border border-border/80 bg-background/70 p-4">
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Execution</p>
              <p className="mt-2 text-lg font-semibold text-foreground">Spread and slippage gates active</p>
            </div>
            <div className="rounded-2xl border border-border/80 bg-background/70 p-4">
              <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Risk</p>
              <p className="mt-2 text-lg font-semibold text-foreground">Kill-switch and reconciliation latch guarded</p>
            </div>
          </div>
        </section>

        <section className="apex-panel apex-fade-up rounded-3xl p-6 sm:p-8">
          <div className="mb-5 flex justify-end">
            <Button
              type="button"
              variant="outline"
              className="rounded-xl"
              onClick={toggleTheme}
              aria-label={
                !mounted
                  ? "Toggle theme"
                  : theme === "dark"
                    ? "Switch to light mode"
                    : "Switch to dark mode"
              }
            >
              {!mounted ? <Moon className="h-4 w-4" /> : theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              {!mounted ? "Theme" : theme === "dark" ? "Light" : "Dark"}
            </Button>
          </div>
          <div className="mb-8 space-y-3">
            <p className="inline-flex items-center gap-2 rounded-full bg-accent px-3 py-1 text-xs font-semibold tracking-wide text-accent-foreground">
              <ShieldCheck className="h-3.5 w-3.5" />
              Secure Operator Login
            </p>
            <h2 className="text-3xl font-semibold tracking-tight text-foreground">Sign in to APEX</h2>
            <p className="text-sm text-muted-foreground">
              Use your assigned admin credentials to access the trading dashboard.
            </p>
            <p className="text-xs text-muted-foreground">
              Need plan details first?{" "}
              <Link href="/pricing" className="font-semibold text-primary hover:underline">
                View pricing
              </Link>
            </p>
          </div>

          <form className="space-y-4" onSubmit={handleLogin} aria-label="APEX admin login form">
            {reason === "session_expired" ? (
              <Alert>
                <AlertTitle>Session expired</AlertTitle>
                <AlertDescription>Please sign in again to continue.</AlertDescription>
              </Alert>
            ) : null}
            {error && (
              <Alert variant="destructive">
                <AlertTitle>Authentication failed</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <div className="relative">
                <User className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="username"
                  aria-label="Username"
                  autoComplete="username"
                  placeholder="admin"
                  className="h-11 rounded-xl pl-10"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="password"
                  aria-label="Password"
                  autoComplete="current-password"
                  type="password"
                  className="h-11 rounded-xl pl-10"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>

            <Button type="submit" className="h-11 w-full rounded-xl text-sm font-semibold" disabled={loading}>
              {loading ? "Authenticating..." : "Log In"}
              {!loading ? <ArrowRight className="h-4 w-4" /> : null}
            </Button>
          </form>
        </section>
      </div>
    </main>
  );
}
