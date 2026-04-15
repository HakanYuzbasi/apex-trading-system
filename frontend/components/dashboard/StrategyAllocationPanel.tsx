"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { LayoutGrid, PieChart as PieChartIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export interface StrategyAllocationData {
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

  // Using theme-consistent colors: Primary blue and Amber
  const COLORS = ["hsl(var(--primary))", "hsl(var(--warning))"];

  if (!allocation || data.length === 0) {
    return (
      <div className="glass-card rounded-2xl p-6 h-full flex flex-col items-center justify-center gap-3 text-muted-foreground opacity-50">
         <LayoutGrid className="h-6 w-6 animate-pulse" />
         <p className="text-[10px] font-black uppercase tracking-widest text-center">Awaiting Strategy Allocation Map...</p>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-2xl p-5 h-full flex flex-col space-y-4 animate-in fade-in duration-500">
      <div className="flex items-center justify-between border-b border-border/20 pb-3">
        <div className="flex items-center gap-2">
          <PieChartIcon size={14} className="text-primary" />
          <h2 className="text-[11px] font-black text-foreground uppercase tracking-tight">Strategy Allocation</h2>
        </div>
        <Badge variant="outline" className="text-[9px] font-bold bg-background/40">NET EXPOSURE</Badge>
      </div>

      <div className="flex-1 flex items-center justify-center min-h-[160px] relative">
        <ResponsiveContainer width="100%" height={160}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={45}
              outerRadius={70}
              paddingAngle={8}
              dataKey="value"
              stroke="none"
              animationBegin={0}
              animationDuration={1500}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} className="hover:opacity-80 transition-opacity cursor-pointer" />
              ))}
            </Pie>
            <Tooltip
              cursor={{ fill: 'transparent' }}
              contentStyle={{ 
                backgroundColor: 'rgba(15, 23, 42, 0.9)', 
                backdropFilter: 'blur(8px)',
                border: '1px solid rgba(255, 255, 255, 0.1)', 
                borderRadius: '12px',
                fontSize: '10px',
                fontWeight: '900',
                textTransform: 'uppercase',
                padding: '8px 12px',
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)'
              }}
              itemStyle={{ color: '#fff' }}
              labelStyle={{ display: 'none' }}
              formatter={(value: any) => [`${(Number(value) || 0).toFixed(1)}%`, "Weight"]}
            />
          </PieChart>
        </ResponsiveContainer>
        
        {/* Center label */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
           <span className="text-[10px] font-black text-muted-foreground uppercase opacity-40">Pods</span>
           <span className="text-sm font-black font-mono text-foreground">{data.length}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 pt-2">
        {data.map((d, i) => (
          <div key={d.name} className="flex flex-col gap-1.5 p-2 rounded-xl bg-background/30 border border-border/10">
            <div className="flex items-center gap-1.5">
               <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }}></span>
               <span className="text-[9px] font-black text-muted-foreground uppercase tracking-widest truncate">{d.name}</span>
            </div>
            <span className="text-xs font-black font-mono text-foreground ml-3">{d.value.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
