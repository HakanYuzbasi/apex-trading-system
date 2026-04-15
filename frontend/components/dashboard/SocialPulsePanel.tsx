"use client";

import { Activity, Zap, Flame, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn, getToneClass } from "@/lib/utils";

export interface SocialPulseData {
  [symbol: string]: number;
}

export default function SocialPulsePanel({ pulse }: { pulse?: SocialPulseData }) {
  if (!pulse || Object.keys(pulse).length === 0) {
    return (
      <div className="glass-card rounded-2xl p-6 h-full flex flex-col items-center justify-center gap-3 text-muted-foreground opacity-50">
        <Activity className="h-6 w-6 animate-pulse" />
        <p className="text-[10px] font-black uppercase tracking-widest text-center">Waiting for social sentiment telemetry...</p>
      </div>
    );
  }

  // Sort by highest absolute z-score
  const entries = Object.entries(pulse).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 5);

  return (
    <div className="glass-card rounded-2xl p-5 h-full space-y-5 animate-in fade-in duration-500">
      <div className="flex items-center justify-between border-b border-border/20 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary animate-pulse" />
          <h2 className="text-[11px] font-black text-foreground uppercase tracking-tight">Social Momentum Pulse</h2>
        </div>
        <Badge variant="outline" className="text-[9px] font-bold bg-background/40">Z-SCORE METER</Badge>
      </div>
      
      <div className="space-y-4">
        {entries.map(([symbol, score]) => {
          const val = Math.max(0, Math.min(10, score + 5)); // Normalizing -5 to +5 range into 0 to 10
          const pct = (val / 10) * 100;
          
          let tone: "positive" | "warning" | "negative" | "neutral" = "neutral";
          if (score > 3.0) tone = "negative"; // Overheated/Pump
          else if (score > 1.5) tone = "warning";
          else if (score < -1.5) tone = "positive"; // FUD/Oversold potential
          
          return (
            <div key={symbol} className="space-y-2 group">
               <div className="flex justify-between items-center">
                 <div className="flex items-center gap-2">
                    <span className="text-[11px] font-black text-foreground group-hover:text-primary transition-colors">{symbol}</span>
                    {score > 3.0 && (
                      <Badge variant="negative" className="text-[8px] h-3.5 px-1 font-black animate-pulse">
                        <Flame size={8} className="mr-0.5" /> PUMP
                      </Badge>
                    )}
                 </div>
                 <span className={cn("text-[10px] font-black font-mono tracking-widest", getToneClass(tone))}>
                   {score > 3.0 ? "CRITICAL_VOL" : `Z=${score.toFixed(2)}`}
                 </span>
               </div>
               <div className="w-full bg-background/40 rounded-full h-1.5 border border-border/10 overflow-hidden relative">
                 <div 
                   className={cn(
                     "h-full rounded-full transition-all duration-700 relative z-10",
                     score > 3.0 ? "bg-negative shadow-[0_0_10px_rgba(239,68,68,0.4)]" : 
                     score > 1.5 ? "bg-warning" : 
                     score < -1.5 ? "bg-primary" : "bg-muted-foreground/40"
                   )} 
                   style={{ width: `${pct}%` }} 
                 />
                 <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
               </div>
            </div>
          );
        })}
      </div>
      
      <div className="pt-2 flex items-center justify-center">
         <p className="text-[9px] font-black text-muted-foreground uppercase tracking-[0.2em] opacity-40">Crowd Intelligence Network Active</p>
      </div>
    </div>
  );
}
