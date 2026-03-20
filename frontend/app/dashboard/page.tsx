"use client";

import Link from "next/link";
import { BarChart3, Bitcoin, TrendingUp, ArrowRight } from "lucide-react";
import { useSessionMetrics } from "@/lib/api";
import { formatCurrency, formatPct } from "@/lib/formatters";
import { SESSION_CONFIG } from "@/lib/constants";

function SessionCard({
  sessionType,
  label,
  description,
  icon: Icon,
  href,
}: {
  sessionType: "core" | "crypto";
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  href: string;
}) {
  const { data: metrics } = useSessionMetrics(sessionType);

  const capital = metrics?.capital ?? SESSION_CONFIG[sessionType].initialCapital;
  const sharpe = metrics?.sharpe_ratio ?? 0;
  const dailyPnl = metrics?.daily_pnl ?? 0;
  const winRate = metrics?.win_rate ?? 0;
  const positions = metrics?.open_positions ?? 0;
  const sharpeTarget = SESSION_CONFIG[sessionType].sharpeTarget;

  return (
    <Link
      href={href}
      className="group flex flex-col gap-4 rounded-2xl border border-border/80 bg-background/70 p-6 transition-all hover:border-primary/40 hover:shadow-lg"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">{label}</h2>
            <p className="text-xs text-muted-foreground">{description}</p>
          </div>
        </div>
        <ArrowRight className="h-5 w-5 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-primary" />
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <p className="text-xs text-muted-foreground">Capital</p>
          <p className="text-sm font-semibold text-foreground">
            {formatCurrency(capital)}
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Sharpe Ratio</p>
          <p
            className={`text-sm font-semibold ${
              sharpe >= sharpeTarget ? "text-positive" : "text-foreground"
            }`}
          >
            {sharpe.toFixed(2)}{" "}
            <span className="text-xs text-muted-foreground">
              / {sharpeTarget.toFixed(1)}
            </span>
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Daily P&L</p>
          <p
            className={`text-sm font-semibold ${
              dailyPnl >= 0 ? "text-positive" : "text-negative"
            }`}
          >
            {formatCurrency(dailyPnl)}
          </p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Win Rate</p>
          <p className="text-sm font-semibold text-foreground">
            {formatPct(winRate)}
          </p>
        </div>
      </div>

      {/* Sharpe progress bar */}
      <div>
        <div className="mb-1 flex justify-between text-xs text-muted-foreground">
          <span>Sharpe Progress</span>
          <span>
            {sharpe.toFixed(2)} / {sharpeTarget.toFixed(1)}
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={`h-full rounded-full transition-all ${
              sharpe >= sharpeTarget ? "bg-positive" : "bg-primary"
            }`}
            style={{
              width: `${Math.min(100, (sharpe / sharpeTarget) * 100)}%`,
            }}
          />
        </div>
      </div>

      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <TrendingUp className="h-3 w-3" />
        <span>{positions} open positions</span>
      </div>
    </Link>
  );
}

export default function DashboardOverview() {
  return (
    <div className="mx-auto max-w-[1200px] px-4 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-foreground">
          Trading Sessions
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Two independent strategies — Core (equities/forex/indices) and Crypto
          Sleeve (toggleable). Both targeting 1.5+ Sharpe.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <SessionCard
          sessionType="core"
          label="Core Strategy"
          description="Equities, indices, and forex — no crypto"
          icon={BarChart3}
          href="/dashboard/core"
        />
        <SessionCard
          sessionType="crypto"
          label="Crypto Sleeve"
          description="Cryptocurrency trading — toggleable"
          icon={Bitcoin}
          href="/dashboard/crypto"
        />
      </div>
    </div>
  );
}
