import * as React from "react"
import { cn } from "@/lib/utils"

interface SegmentedControlProps {
  options: { label: string; value: string; icon?: React.ReactNode }[]
  value: string
  onChange: (value: string) => void
  className?: string
}

export function SegmentedControl({ options, value, onChange, className }: SegmentedControlProps) {
  return (
    <div className={cn("flex gap-1 rounded-xl border border-border p-1 bg-muted/20", className)}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={cn(
            "flex-1 flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-sm font-semibold transition-all duration-200",
            value === option.value
              ? "bg-primary text-primary-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
          )}
        >
          {option.icon}
          {option.label}
        </button>
      ))}
    </div>
  )
}
