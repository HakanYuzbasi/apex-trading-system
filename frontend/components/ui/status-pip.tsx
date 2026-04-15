import * as React from "react"
import { cn } from "@/lib/utils"

interface StatusPipProps extends React.HTMLAttributes<HTMLSpanElement> {
  active?: boolean
  variant?: "emerald" | "amber" | "rose" | "zinc"
}

export function StatusPip({ active, variant, className, ...props }: StatusPipProps) {
  const colorClass = active 
    ? {
        emerald: "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.7)]",
        amber: "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.7)]",
        rose: "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.7)]",
        zinc: "bg-zinc-500",
      }[variant || "emerald"]
    : "bg-zinc-500/50"

  return (
    <span
      className={cn(
        "inline-block h-2 w-2 rounded-full transition-all duration-300",
        colorClass,
        className
      )}
      {...props}
    />
  )
}
