"use client";

import React from "react";
import { ShieldAlert, ShieldCheck, Info, AlertTriangle } from "lucide-react";

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
    <div className="bg-card/50 backdrop-blur-md rounded-xl border border-white/10 p-5 h-full overflow-hidden flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className={`p-2 rounded-lg ${isHealthy ? 'bg-positive/10 text-positive' : 'bg-warning/10 text-warning'}`}>
            {isHealthy ? <ShieldCheck size={20} /> : <ShieldAlert size={20} />}
          </div>
          <div>
            <h3 className="text-sm font-bold text-foreground">Warden's Eye</h3>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
              Sentiment Health
            </p>
          </div>
        </div>
        <div className="flex gap-1">
            <span className={`px-2 py-1 rounded text-[10px] font-bold ${isHealthy ? 'bg-positive/20 text-positive' : 'bg-warning/20 text-warning'}`}>
              {isHealthy ? "OPTIMAL" : "VETO ACTIVE"}
            </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
        {isHealthy ? (
          <div className="flex flex-col items-center justify-center h-full opacity-40 py-8 text-center">
            <ShieldCheck size={48} className="mb-3 text-positive" strokeWidth={1} />
            <p className="text-xs font-medium">No structural threats detected.</p>
            <p className="text-[10px] mt-1">Active universe is cleared for entry.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {Object.entries(sentiment_health).map(([symbol, detail]) => (
              <div key={symbol} className="p-3 rounded-lg bg-white/5 border border-white/5 hover:border-warning/30 transition-colors group">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-bold text-warning">{symbol}</span>
                  <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                    <AlertTriangle size={10} /> 24H VETO
                  </span>
                </div>
                <p className="text-[11px] leading-relaxed text-foreground/80 italic mb-2 group-hover:text-foreground transition-colors">
                  "{detail.headline}"
                </p>
                <div className="flex items-center justify-between text-[9px] text-muted-foreground">
                  <span>Detected: {new Date(detail.detected_at).toLocaleTimeString()}</span>
                  <span>Expires in: {Math.max(0, Math.round((detail.expires_at - Date.now()/1000)/3600))}h</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="mt-4 pt-3 border-t border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground font-medium">
          <Info size={12} />
          <span>Category B Filter Active</span>
        </div>
        <div className="text-[10px] text-muted-foreground font-medium">
          Source: Alpaca + Gemini 3
        </div>
      </div>
    </div>
  );
}
