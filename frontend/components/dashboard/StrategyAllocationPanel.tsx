"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

interface StrategyAllocationData {
  KalmanPairs?: number;
  BreakoutPod?: number;
}

export default function StrategyAllocationPanel({ allocation }: { allocation?: StrategyAllocationData }) {
  const pairs = allocation?.KalmanPairs ?? 0.0;
  const breakOut = allocation?.BreakoutPod ?? 0.0;

  const data = [
    { name: "Kalman Pairs", value: pairs * 100 },
    { name: "Breakout Pod", value: breakOut * 100 },
  ].filter((d) => d.value > 0);

  const COLORS = ["#3b82f6", "#f59e0b"]; // Blue for Pairs, Amber for Breakout

  if (!allocation || data.length === 0) {
    return (
      <div className="rounded-xl border border-border/80 bg-background/70 p-4">
         <h2 className="text-sm font-semibold text-foreground mb-2">Strategy Allocation</h2>
         <p className="text-xs text-muted-foreground">Waiting for state updates...</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border/80 bg-background/70 p-4 h-full flex flex-col">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-foreground">Strategy Allocation</h2>
      </div>
      <div className="flex-1 flex items-center justify-center min-h-[150px]">
        <ResponsiveContainer width="100%" height={150}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={40}
              outerRadius={65}
              paddingAngle={5}
              dataKey="value"
              stroke="none"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value: any) => [`${(Number(value) || 0).toFixed(1)}%`, "Allocation"]}
              contentStyle={{ borderRadius: "8px", border: "none", fontSize: "12px", backgroundColor: "#1e293b", color: "#f8fafc" }}
              itemStyle={{ color: "#f8fafc" }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex justify-center gap-4 text-[11px]">
        {data.map((d, i) => (
          <div key={d.name} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }}></span>
            <span className="text-muted-foreground">{d.name} ({d.value.toFixed(1)}%)</span>
          </div>
        ))}
      </div>
    </div>
  );
}
