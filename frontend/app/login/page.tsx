"use client";

import { Suspense, useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Lock, LogIn, Server, Key, AlertCircle, RefreshCw } from "lucide-react";
import { useAuthContext } from "@/components/auth/AuthProvider";

import { useTheme } from "@/components/theme/ThemeProvider";
import { Button } from "@/components/ui/button";

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login } = useAuthContext();
  const { theme, setTheme } = useTheme();

  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Example "connection" status for effect
  const [coreStatus, setCoreStatus] = useState<"connecting" | "ready" | "error">("connecting");

  useEffect(() => {
    // Simulate initial connection check for the "Institutional" feel
    const timer = setTimeout(() => {
      setCoreStatus("ready");
    }, 1500);
    return () => clearTimeout(timer);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const success = await login("admin", password);
      if (success) {
        // Successful login, redirect to where they wanted to go, or /dashboard
        const returnUrl = searchParams.get("returnUrl") || "/dashboard";
        router.push(returnUrl);
      } else {
        setError("Invalid master key provided.");
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred during authentication.");
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

      {coreStatus === "connecting" && (
        <div className="mb-6 flex items-center justify-center gap-2 rounded-lg border border-border/50 bg-secondary/50 p-3 text-sm text-muted-foreground">
          <RefreshCw className="h-4 w-4 animate-spin text-primary" />
          <span>Establishing secure connection to core...</span>
        </div>
      )}

      {coreStatus === "ready" && (
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-2">
            <label
              htmlFor="master-key"
              className="flex items-center gap-2 text-sm font-medium text-foreground"
            >
              <Key className="h-4 w-4 text-muted-foreground" />
              Master Key
            </label>
            <div className="relative">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                <Lock className="h-4 w-4 text-muted-foreground" />
              </div>
              <input
                id="master-key"
                type="password"
                required
                className="block w-full rounded-md border border-input bg-background py-2 pl-10 text-sm font-medium text-foreground shadow-inner focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                placeholder="Enter standard symmetric key"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isLoading}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Required for read-only cockpit and policy overrides.
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
      )}
    </>
  );
}

export default function LoginPage() {
  const { theme, setTheme } = useTheme();

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
