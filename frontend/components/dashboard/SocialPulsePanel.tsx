"use client";

import { Activity } from "lucide-react";

export interface SocialPulseData {
  [symbol: string]: number;
}

export default function SocialPulsePanel({ pulse }: { pulse?: SocialPulseData }) {
  if (!pulse || Object.keys(pulse).length === 0) {
    return (
      <div className="rounded-xl border border-border/80 bg-background/70 p-4">
         <div className="flex items-center gap-2 mb-2">
           <Activity className="h-4 w-4 text-muted-foreground" />
           <h2 className="text-sm font-semibold text-foreground">Social Momentum Pulse</h2>
         </div>
         <p className="text-xs text-muted-foreground">Waiting for sentiment data...</p>
      </div>
    );
  }

  // Sort by highest absolute z-score
  const entries = Object.entries(pulse).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 5);

  return (
    <div className="rounded-xl border border-border/80 bg-background/70 p-4 h-full">
      <div className="flex items-center gap-2 mb-3">
        <Activity className="h-4 w-4 text-purple-400" />
        <h2 className="text-sm font-semibold text-foreground">Social Momentum Pulse</h2>
      </div>
      
      <div className="flex flex-col gap-2">
        {entries.map(([symbol, score]) => {
          const val = Math.max(0, Math.min(10, score + 5)); // Normalizing -5 to +5 range into 0 to 10
          const pct = (val / 10) * 100;
          
          let color = "bg-primary";
          if (score > 3.0) color = "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]";
          else if (score > 1.5) color = "bg-amber-400";
          else if (score < -1.5) color = "bg-blue-400";
          
          return (
            <div key={symbol} className="flex flex-col gap-1">
               <div className="flex justify-between items-center text-xs">
                 <span className="font-semibold text-foreground">{symbol}</span>
                 <span className="text-muted-foreground font-mono">
                   {score > 3.0 ? "PUMP DETECTED" : `z=${score.toFixed(2)}`}
                 </span>
               </div>
               <div className="w-full bg-secondary/50 rounded-full h-1.5 border border-border/40 overflow-hidden">
                 <div className={`h-full rounded-full transition-all duration-500 ${color}`} style={{ width: `${pct}%` }}></div>
               </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
