"use client";

import { Suspense, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  AlertCircle,
  Eye,
  EyeOff,
  Key,
  Lock,
  LogIn,
  RefreshCw,
  Server,
  User,
} from "lucide-react";
import { useAuthContext } from "@/components/auth/AuthProvider";

import { Button } from "@/components/ui/button";

function formatLoginError(rawMessage: string): string {
  const message = rawMessage.trim();
  const lower = message.toLowerCase();
  if (lower.includes("invalid") || lower.includes("incorrect")) {
    return "Invalid username or password.";
  }
  if (lower.includes("unreachable") || lower.includes("failed to fetch") || lower.includes("could not reach")) {
    return "Authentication service is unavailable. Verify backend health and try again.";
  }
  return message || "Authentication failed.";
}

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login } = useAuthContext();

  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const loginReason = searchParams.get("reason");
  const reasonBanner = useMemo(() => {
    if (loginReason === "session_expired") {
      return "Session expired. Please authenticate again.";
    }
    if (loginReason === "unauthorized") {
      return "Access denied for your current session. Re-authenticate to continue.";
    }
    if (loginReason === "backend_unreachable") {
      return "Backend service is currently unreachable.";
    }
    return "";
  }, [loginReason]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const result = await login(username.trim(), password);
      if (result.ok) {
        const requestedReturnUrl = searchParams.get("returnUrl");
        const returnUrl = requestedReturnUrl && requestedReturnUrl.startsWith("/") ? requestedReturnUrl : "/dashboard";
        router.push(returnUrl);
      } else {
        setError(formatLoginError(result.error || ""));
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "An unexpected error occurred during authentication.";
      setError(formatLoginError(message));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <div className="mb-8 p-6 text-center">
        <Server className="mx-auto h-12 w-12 text-primary drop-shadow-[0_0_15px_rgba(var(--primary),0.3)]" />
        <h1 className="mt-4 text-3xl font-bold tracking-tight text-foreground">
          Apex Engine
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Algorithmic Trading & Risk Governance
        </p>
      </div>

      {reasonBanner ? (
        <div className="mb-6 flex items-center gap-2 rounded-lg border border-amber-300/60 bg-amber-50/70 p-3 text-sm text-amber-900 dark:border-amber-700/60 dark:bg-amber-950/40 dark:text-amber-200">
          <AlertCircle className="h-4 w-4" />
          <span>{reasonBanner}</span>
        </div>
      ) : null}

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-2">
          <label
            htmlFor="username"
            className="flex items-center gap-2 text-sm font-medium text-foreground"
          >
            <User className="h-4 w-4 text-muted-foreground" />
            Username
          </label>
          <div className="relative">
            <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
              <User className="h-4 w-4 text-muted-foreground" />
            </div>
            <input
              id="username"
              type="text"
              required
              autoComplete="username"
              className="block w-full rounded-md border border-input bg-background py-2 pl-10 text-sm font-medium text-foreground shadow-inner focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              placeholder="admin"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={isLoading}
            />
          </div>
        </div>

        <div className="space-y-2">
          <label
            htmlFor="password"
            className="flex items-center gap-2 text-sm font-medium text-foreground"
          >
            <Key className="h-4 w-4 text-muted-foreground" />
            Password / Master Key
          </label>
          <div className="relative">
            <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
              <Lock className="h-4 w-4 text-muted-foreground" />
            </div>
            <input
              id="password"
              type={showPassword ? "text" : "password"}
              required
              autoComplete="current-password"
              className="block w-full rounded-md border border-input bg-background py-2 pl-10 pr-10 text-sm font-medium text-foreground shadow-inner focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
              placeholder="Enter admin password (APEX_ADMIN_PASSWORD)"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={isLoading}
            />
            <button
              type="button"
              className="absolute inset-y-0 right-0 flex items-center px-3 text-muted-foreground transition hover:text-foreground"
              onClick={() => setShowPassword((value) => !value)}
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
          <p className="text-xs text-muted-foreground">
            Use your account username and the configured admin password for cockpit access.
          </p>
        </div>

        {error && (
          <div className="flex items-center gap-2 rounded-md border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="h-4 w-4" />
            <span>{error}</span>
          </div>
        )}

        <Button
          type="submit"
          className="w-full shadow-[0_0_10px_rgba(var(--primary),0.2)] transition-shadow hover:shadow-[0_0_20px_rgba(var(--primary),0.4)]"
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
              Validating...
            </>
          ) : (
            <>
              <LogIn className="mr-2 h-4 w-4" />
              Authenticate
            </>
          )}
        </Button>
      </form>
    </>
  );
}

export default function LoginPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background/95 p-4 antialiased selection:bg-primary/20">
      <div className="pointer-events-none absolute inset-0 overflow-hidden opacity-20">
        <div className="absolute -left-[20%] -top-[20%] h-[70%] w-[50%] rounded-full bg-primary/20 blur-[120px]" />
        <div className="absolute -bottom-[20%] -right-[20%] h-[70%] w-[50%] rounded-full bg-secondary/30 blur-[120px]" />
      </div>

      <div className="relative z-10 w-full max-w-md overflow-hidden rounded-2xl border border-border/60 bg-background/80 shadow-2xl backdrop-blur-xl">
        <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
        <div className="p-8">
          <Suspense fallback={
            <div className="flex items-center justify-center p-6 text-sm text-muted-foreground">
              <RefreshCw className="h-4 w-4 animate-spin mr-2" />
              Loading login context...
            </div>
          }>
            <LoginForm />
          </Suspense>
        </div>
      </div>
    </div>
  );
}
