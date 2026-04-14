"use client";

import { LoaderCircle } from "lucide-react";

type LoadingSpinnerProps = {
  label?: string;
  detail?: string;
  className?: string;
};

export function LoadingSpinner({
  label = "Loading live telemetry",
  detail,
  className = "",
}: LoadingSpinnerProps) {
  return (
    <div
      className={[
        "flex min-h-[220px] flex-col items-center justify-center gap-3 rounded-2xl border border-zinc-800/80 bg-zinc-950/90 px-6 py-10 text-center text-zinc-200 shadow-[0_20px_50px_rgba(0,0,0,0.35)]",
        className,
      ].join(" ")}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full border border-cyan-500/30 bg-cyan-500/10">
        <LoaderCircle className="h-5 w-5 animate-spin text-cyan-300" />
      </div>
      <div className="space-y-1">
        <p className="text-sm font-semibold tracking-wide text-zinc-100">{label}</p>
        <p className="text-xs text-zinc-400">
          {detail ?? "Waiting for the latest state snapshot from the trading engine."}
        </p>
      </div>
    </div>
  );
}
