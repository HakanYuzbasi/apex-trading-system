"use client";

import React from "react";
import { ShieldAlert, ShieldCheck, Info, AlertTriangle, Eye, Clock } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export interface VetoDetail {
  expires_at: number;
  headline: string;
  detected_at: string;
}

interface SentimentHealthProps {
  sentiment_health?: Record<string, VetoDetail>;
}

export default function SentimentHealthPanel({ sentiment_health = {} }: SentimentHealthProps) {
  const vetoedSymbols = Object.keys(sentiment_health);
  const isHealthy = vetoedSymbols.length === 0;

  return (
    <div className="glass-card rounded-2xl p-6 h-full overflow-hidden flex flex-col animate-in fade-in duration-500">
      <div className="flex items-center justify-between mb-6 border-b border-border/40 pb-4">
        <div className="flex items-center gap-3">
          <div className={cn(
            "p-2.5 rounded-xl transition-all duration-300 shadow-inner",
            isHealthy ? 'bg-positive/10 text-positive' : 'bg-warning/10 text-warning ring-1 ring-warning/30'
          )}>
            {isHealthy ? <ShieldCheck size={20} /> : <ShieldAlert size={20} className="animate-pulse" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
               <Eye size={12} className="text-primary" />
               <h3 className="text-sm font-black text-foreground uppercase tracking-tight">Warden's Eye</h3>
            </div>
            <p className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground font-black">
              Sentiment Guard
            </p>
          </div>
        </div>
        <Badge 
          variant={isHealthy ? "positive" : "warning"} 
          className="text-[10px] font-black h-6 px-3 tracking-widest"
        >
          {isHealthy ? "SYSTEM_OPTIMAL" : "VETO_ACTIVE"}
        </Badge>
      </div>

      <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
        {isHealthy ? (
          <div className="flex flex-col items-center justify-center h-full py-12 text-center space-y-4">
            <div className="relative">
               <ShieldCheck size={64} className="text-positive opacity-10" strokeWidth={1} />
               <ShieldCheck size={32} className="text-positive absolute inset-0 m-auto animate-in zoom-in duration-700" strokeWidth={2} />
            </div>
            <div className="space-y-1">
              <p className="text-xs font-bold text-foreground/60 uppercase tracking-widest">No structural threats</p>
              <p className="text-[10px] text-muted-foreground font-medium">Global universe cleared for deployment.</p>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {Object.entries(sentiment_health).map(([symbol, detail]) => (
              <div key={symbol} className="p-4 rounded-2xl bg-background/30 border border-border/20 hover:border-warning/40 transition-all group relative overflow-hidden">
                <div className="absolute top-0 right-0 p-1 opacity-5 group-hover:opacity-10 transition-opacity">
                   <AlertTriangle size={48} className="text-warning" />
                </div>
                
                <div className="flex items-center justify-between mb-3">
                  <Badge variant="warning" className="font-mono text-[11px] font-black bg-background/60">
                    {symbol}
                  </Badge>
                  <div className="flex items-center gap-1.5 text-[10px] font-bold text-warning uppercase">
                    <AlertTriangle size={12} className="mb-0.5" />
                    <span>24H Veto</span>
                  </div>
                </div>
                
                <p className="text-[11px] leading-relaxed text-foreground font-bold italic mb-4 border-l-2 border-warning/30 pl-3">
                  "{detail.headline}"
                </p>
                
                <div className="flex items-center justify-between text-[10px] font-bold text-muted-foreground font-mono">
                  <div className="flex items-center gap-1.5">
                    <Clock size={12} />
                    <span>{new Date(detail.detected_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                  <Badge variant="outline" className="text-[9px] h-5 bg-background/20 border-border/40">
                    EXP: {Math.max(0, Math.round((detail.expires_at - Date.now()/1000)/3600))}H
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="mt-6 pt-4 border-t border-border/30 flex items-center justify-between">
        <div className="flex items-center gap-2 text-[10px] font-bold text-muted-foreground uppercase tracking-widest bg-muted/20 px-3 py-1.5 rounded-full">
          <Info size={12} className="text-primary" />
          <span>Category B Filter</span>
        </div>
        <div className="text-[9px] font-bold text-muted-foreground/60 uppercase text-right">
          Provider: Neural Engine v4
        </div>
      </div>
    </div>
  );
}
