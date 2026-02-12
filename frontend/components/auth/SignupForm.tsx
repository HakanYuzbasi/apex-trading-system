"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthContext } from "./AuthProvider";
import { cn } from "@/lib/utils";

interface SignupFormProps {
  onSwitchToLogin?: () => void;
  className?: string;
}

export function SignupForm({ onSwitchToLogin, className }: SignupFormProps) {
  const { register, isLoading } = useAuthContext();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }
    const result = await register(username.trim(), email.trim(), password);
    if (result.ok) {
      router.push("/");
      return;
    }
    setError(result.error ?? "Registration failed");
  }

  return (
    <form onSubmit={handleSubmit} className={cn("space-y-4", className)}>
      {error && (
        <div role="alert" aria-live="assertive" className="rounded-md bg-red-500/10 border border-red-500/30 text-red-400 text-sm px-3 py-2">
          {error}
        </div>
      )}
      <div>
        <label htmlFor="signup-username" className="block text-sm font-medium text-foreground/80 mb-1">
          Username
        </label>
        <input
          id="signup-username"
          type="text"
          autoComplete="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full rounded-lg border border-border bg-background/80 px-3 py-2 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          placeholder="johndoe"
          minLength={3}
          required
        />
      </div>
      <div>
        <label htmlFor="signup-email" className="block text-sm font-medium text-foreground/80 mb-1">
          Email
        </label>
        <input
          id="signup-email"
          type="email"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full rounded-lg border border-border bg-background/80 px-3 py-2 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          placeholder="you@example.com"
          required
        />
      </div>
      <div>
        <label htmlFor="signup-password" className="block text-sm font-medium text-foreground/80 mb-1">
          Password
        </label>
        <input
          id="signup-password"
          type="password"
          autoComplete="new-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full rounded-lg border border-border bg-background/80 px-3 py-2 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          placeholder="At least 8 characters"
          minLength={8}
          required
        />
      </div>
      <button
        type="submit"
        disabled={isLoading}
        className="w-full rounded-lg bg-primary text-primary-foreground font-medium py-2 px-4 hover:opacity-90 disabled:opacity-50 transition"
      >
        {isLoading ? "Creating account…" : "Create account"}
      </button>
      {onSwitchToLogin && (
        <p className="text-center text-sm text-muted-foreground">
          Already have an account?{" "}
          <button type="button" onClick={onSwitchToLogin} className="text-primary hover:underline">
            Sign in
          </button>
        </p>
      )}
    </form>
  );
}
