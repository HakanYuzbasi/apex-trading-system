"use client";

import { useAuthContext } from "@/components/auth/AuthProvider";
import Link from "next/link";

export default function SettingsPage() {
  const { user, logout, isAuthenticated } = useAuthContext();

  if (!isAuthenticated || !user) {
    return (
      <main className="min-h-screen flex flex-col items-center justify-center p-4">
        <p className="text-muted-foreground">You must be signed in to view settings.</p>
        <Link href="/login" className="mt-2 text-primary hover:underline">Sign in</Link>
      </main>
    );
  }

  return (
    <main className="min-h-screen p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-semibold text-foreground mb-6">Account & subscription</h1>
      <div className="rounded-lg border border-border bg-card/50 p-4 space-y-3">
        <p><span className="text-muted-foreground">Username:</span> {user.username}</p>
        {user.email && <p><span className="text-muted-foreground">Email:</span> {user.email}</p>}
        <p><span className="text-muted-foreground">Tier:</span> <span className="capitalize">{user.tier}</span></p>
      </div>
      <div className="mt-6 flex gap-3">
        <button
          onClick={() => logout()}
          className="rounded-lg border border-border px-4 py-2 text-foreground hover:bg-muted/50 transition"
        >
          Sign out
        </button>
        <Link href="/" className="rounded-lg bg-primary text-primary-foreground px-4 py-2 hover:opacity-90 transition">
          Dashboard
        </Link>
      </div>
    </main>
  );
}
