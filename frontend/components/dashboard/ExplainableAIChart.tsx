"use client";

import { useMemo } from "react";
import { BrainCircuit } from "lucide-react";

export type ExplainableAIChartProps = {
  shapData: Record<string, number> | null;
  symbol?: string;
};

export default function ExplainableAIChart({ shapData, symbol }: ExplainableAIChartProps) {
  const chartData = useMemo(() => {
    if (!shapData) return [];
    // Convert object to array, sort by absolute impact (largest first)
    return Object.entries(shapData)
      .map(([feature, value]) => ({ feature, value }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 5); // Keep top 5 drivers
  }, [shapData]);

  if (!chartData || chartData.length === 0) {
    return (
      <div className="flex h-[180px] w-full flex-col items-center justify-center rounded-2xl border border-border/80 bg-background/70 text-muted-foreground p-5 shadow-sm">
        <BrainCircuit className="mb-2 h-6 w-6 opacity-50 text-primary" />
        <p className="text-xs font-medium">Awaiting live SHAP explanations...</p>
      </div>
    );
  }

  // Calculate max absolute value to scale the bars correctly
  const maxAbsValue = Math.max(...chartData.map((d) => Math.abs(d.value)), 0.01);

  return (
    <div className="bg-card border border-border rounded-2xl p-5 shadow-sm w-full h-full flex flex-col justify-center">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BrainCircuit className="h-5 w-5 text-primary" />
          <div>
            <h3 className="text-sm font-semibold text-foreground">
              AI Logic Explanation {symbol ? `(${symbol})` : ""}
            </h3>
            <p className="text-[11px] text-muted-foreground mt-0.5">Real-time SHAP feature attribution</p>
          </div>
        </div>
        <span className="rounded-full bg-secondary px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-secondary-foreground">
          Top Drivers
        </span>
      </div>

      <div className="space-y-3 mt-2 flex-grow flex flex-col justify-center">
        {chartData.map((item) => {
          const isPositive = item.value >= 0;
          // Calculate percentage width (0 to 100%)
          const widthPct = Math.min((Math.abs(item.value) / maxAbsValue) * 100, 100);

          return (
            <div key={item.feature} className="relative flex items-center text-xs">
              <div className="w-[35%] truncate pr-3 text-right font-medium text-muted-foreground" title={item.feature}>
                {item.feature}
              </div>

              <div className="relative flex h-6 flex-grow items-center">
                {/* Center line */}
                <div className="absolute left-0 top-0 bottom-0 w-[2px] bg-border/80 z-10" />
                
                {/* The Bar */}
                <div
                  className={`h-4 rounded-r-sm transition-all duration-700 ease-out z-0 ${
                    isPositive ? "bg-positive/80 shadow-[0_0_10px_rgba(34,197,94,0.2)]" : "bg-negative/80 shadow-[0_0_10px_rgba(239,68,68,0.2)]"
                  }`}
                  style={{ width: `${widthPct}%` }}
                />
                
                {/* Value Label */}
                <span className={`ml-3 text-[11px] font-bold tracking-wide ${isPositive ? "text-positive" : "text-negative"}`}>
                  {isPositive ? "+" : ""}{item.value.toFixed(3)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
