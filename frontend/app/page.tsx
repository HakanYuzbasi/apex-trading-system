"use client";

import Dashboard from '@/components/Dashboard';
import { AuthGuard } from '@/components/auth/AuthGuard';

export default function Home() {
  return (
    <AuthGuard>
      <main className="min-h-screen bg-transparent relative z-10">
        <Dashboard />
      </main>
    </AuthGuard>
  );
}
