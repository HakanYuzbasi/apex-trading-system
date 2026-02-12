"use client";

import { useState } from "react";
import { LoginForm } from "@/components/auth/LoginForm";
import { SignupForm } from "@/components/auth/SignupForm";
import Link from "next/link";

type Tab = "login" | "signup";

export default function LoginPage() {
  const [tab, setTab] = useState<Tab>("login");

  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-background p-4">
      <div className="w-full max-w-sm rounded-xl border border-border bg-card/50 p-6 shadow-lg">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-semibold text-foreground">APEX Terminal</h1>
          <p className="text-sm text-muted-foreground mt-1">Sign in or create an account</p>
        </div>
        {tab === "login" ? (
          <LoginForm onSwitchToSignup={() => setTab("signup")} />
        ) : (
          <SignupForm onSwitchToLogin={() => setTab("login")} />
        )}
      </div>
      <p className="mt-4 text-sm text-muted-foreground">
        <Link href="/" className="text-primary hover:underline">Back to dashboard</Link>
      </p>
    </main>
  );
}
