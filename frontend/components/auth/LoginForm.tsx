"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthContext } from "./AuthProvider";
import { cn } from "@/lib/utils";

interface LoginFormProps {
  onSwitchToSignup?: () => void;
  className?: string;
}

export function LoginForm({ onSwitchToSignup, className }: LoginFormProps) {
  const { login, isLoading } = useAuthContext();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const result = await login(username.trim(), password);
    if (result.ok) {
      router.push("/");
      return;
    }
    setError(result.error ?? "Login failed");
  }

  return (
    <form onSubmit={handleSubmit} className={cn("space-y-4", className)}>
      {error && (
        <div role="alert" aria-live="assertive" className="rounded-md bg-red-500/10 border border-red-500/30 text-red-400 text-sm px-3 py-2">
          {error}
        </div>
      )}
      <div>
        <label htmlFor="login-username" className="block text-sm font-medium text-foreground/80 mb-1">
          Username or email
        </label>
        <input
          id="login-username"
          type="text"
          autoComplete="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full rounded-lg border border-border bg-background/80 px-3 py-2 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          placeholder="you@example.com"
          required
        />
      </div>
      <div>
        <label htmlFor="login-password" className="block text-sm font-medium text-foreground/80 mb-1">
          Password
        </label>
        <input
          id="login-password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full rounded-lg border border-border bg-background/80 px-3 py-2 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          required
        />
      </div>
      <button
        type="submit"
        disabled={isLoading}
        className="w-full rounded-lg bg-primary text-primary-foreground font-medium py-2 px-4 hover:opacity-90 disabled:opacity-50 transition"
      >
        {isLoading ? "Signing in…" : "Sign in"}
      </button>
      {onSwitchToSignup && (
        <p className="text-center text-sm text-muted-foreground">
          No account?{" "}
          <button type="button" onClick={onSwitchToSignup} className="text-primary hover:underline">
            Sign up
          </button>
        </p>
      )}
    </form>
  );
}
