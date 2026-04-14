"use client";

import { AlertTriangle, RefreshCw } from "lucide-react";

type ErrorStateProps = {
  title?: string;
  message?: string;
  actionLabel?: string;
  onRetry?: () => void;
  className?: string;
};

export function ErrorState({
  title = "Telemetry unavailable",
  message = "The live dashboard could not parse the latest backend payload.",
  actionLabel = "Retry",
  onRetry,
  className = "",
}: ErrorStateProps) {
  return (
    <div
      className={[
        "flex min-h-[220px] flex-col items-center justify-center gap-4 rounded-2xl border border-rose-500/30 bg-zinc-950/95 px-6 py-10 text-center text-zinc-100 shadow-[0_20px_50px_rgba(0,0,0,0.4)]",
        className,
      ].join(" ")}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full border border-rose-500/30 bg-rose-500/10">
        <AlertTriangle className="h-5 w-5 text-rose-300" />
      </div>
      <div className="space-y-1.5">
        <p className="text-sm font-semibold tracking-wide text-zinc-100">{title}</p>
        <p className="max-w-xl text-xs leading-5 text-zinc-400">{message}</p>
      </div>
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex items-center gap-2 rounded-full border border-cyan-500/40 bg-cyan-500/10 px-4 py-2 text-xs font-semibold text-cyan-200 transition-colors hover:bg-cyan-500/20"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          {actionLabel}
        </button>
      ) : null}
    </div>
  );
}
