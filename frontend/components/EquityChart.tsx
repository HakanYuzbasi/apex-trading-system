"use client";

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

interface EquityPoint {
  time: string;
  equity: number;
}

interface EquityChartProps {
  data: EquityPoint[];
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-lg shadow-2xl backdrop-blur-sm">
        <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-1">{payload[0].payload.time}</p>
        <p className="text-sm font-mono font-bold text-emerald-400">
          ${payload[0].value.toLocaleString(undefined, { minimumFractionDigits: 2 })}
        </p>
      </div>
    );
  }
  return null;
};

export default function EquityChart({ data }: EquityChartProps) {
  // Calculate dynamic domain to keep the line centered
  const equities = data.map(d => d.equity);
  const minEquity = Math.min(...equities);
  const maxEquity = Math.max(...equities);
  const padding = (maxEquity - minEquity) * 0.1 || 10;

  return (
    <div className="w-full h-[300px] glass-card rounded-2xl border border-slate-800 bg-slate-900/20 p-4 relative overflow-hidden">
      {/* Background Decor */}
      <div className="absolute top-0 right-0 p-4 opacity-10">
        <LineChart width={40} height={40} data={[{v:1},{v:2},{v:3}]}>
          <Line type="monotone" dataKey="v" stroke="#10b981" strokeWidth={2} dot={false} />
        </LineChart>
      </div>
      
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Real-Time Equity Curve</h3>
        </div>
        <span className="text-[10px] font-mono text-slate-600 italic">60s Window</span>
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.1}/>
              <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke="#1e293b" strokeDasharray="3 3" opacity={0.5} />
          <XAxis 
            dataKey="time" 
            hide 
          />
          <YAxis 
            domain={[minEquity - padding, maxEquity + padding]} 
            hide 
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="equity"
            stroke="#10b981"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#colorEquity)"
            isAnimationActive={false} // Disable for sub-second smooth flow
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
