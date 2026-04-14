"use client";

import { useEffect, useState } from "react";
import { AlertCircle, Activity } from "lucide-react";

export function VPINPanel() {
  const [vpin, setVpin] = useState<number>(0.0);

  // In a real environment, this would poll the execution engine
  // For the dashboard v9, we will mock the poll
  useEffect(() => {
    const interval = setInterval(() => {
      // simulate vpin bobbling around 0.5 with sudden spikes
      setVpin((prev) => {
        const jump = Math.random();
        if (jump > 0.95) return 0.85; // Toxic flow!
        if (jump < 0.05) return 0.1;
        return Math.max(0, Math.min(1.0, prev + (Math.random() - 0.5) * 0.1));
      });
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const isToxic = vpin > 0.8;
  const percentage = Math.min(100, Math.max(0, vpin * 100));

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border p-6 transition-all duration-500 ${
        isToxic
          ? "border-red-500/50 bg-red-500/10 shadow-[0_0_20px_rgba(239,68,68,0.2)]"
          : "border-border/80 bg-background/70 hover:border-primary/40 hover:shadow-lg"
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div
            className={`flex h-10 w-10 items-center justify-center rounded-xl ${
              isToxic ? "bg-red-500/20 text-red-500" : "bg-primary/10 text-primary"
            }`}
          >
            {isToxic ? <AlertCircle className="h-5 w-5" /> : <Activity className="h-5 w-5" />}
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">Microstructure View</h2>
            <p className="text-xs text-muted-foreground">Volume-synchronized Probability of Informed Trading</p>
          </div>
        </div>
        <div className="text-right">
          <p className={`text-2xl font-bold ${isToxic ? "text-red-500" : "text-foreground"}`}>
            {vpin.toFixed(3)}
          </p>
          <p className="text-xs text-muted-foreground">vPIN</p>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <div className="mb-1 flex justify-between text-xs font-medium">
            <span className={isToxic ? "text-red-400" : "text-muted-foreground"}>
              {isToxic ? "DEEP SHADOW VETO ACTIVE" : "Market Flow Normal"}
            </span>
            <span className={isToxic ? "text-red-500" : "text-muted-foreground"}>
              {percentage.toFixed(1)}%
            </span>
          </div>
          <div className="h-3 w-full overflow-hidden rounded-full bg-muted/50">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                isToxic ? "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]" : "bg-primary"
              }`}
              style={{ width: `${percentage}%` }}
            />
          </div>
        </div>

        <p className="text-xs text-muted-foreground">
          {isToxic 
            ? "WARNING: Massive directional imbalance detected! Sniper limit orders have been withdrawn to avoid adverse selection flash crashes."
            : "Liquidity profiles are balanced. OBI Sniper is providing standing limit overlays."}
        </p>
      </div>
    </div>
  );
}
