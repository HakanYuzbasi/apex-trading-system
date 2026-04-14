"use client";

import { useEffect, useState } from "react";
import { Activity, LayoutGrid, Target, ShieldAlert } from "lucide-react";

export function LiquidityPanel() {
  const [data, setData] = useState({
    rebates: 1420.50,
    fees: 310.20,
    netAlpha: 1110.30,
    venues: [
      { name: "Alpaca", btcDepth: 42, ethDepth: 38, vpin: 0.2 },
      { name: "Coinbase", btcDepth: 78, ethDepth: 65, vpin: 0.5 },
      { name: "Kraken", btcDepth: 85, ethDepth: 90, vpin: 0.9 }, // High toxicity!
    ]
  });

  // Mock live updates
  useEffect(() => {
    const timer = setInterval(() => {
      setData((prev) => ({
        ...prev,
        venues: prev.venues.map(v => ({
          ...v,
          btcDepth: Math.max(10, Math.min(100, v.btcDepth + (Math.random() - 0.5) * 10)),
          ethDepth: Math.max(10, Math.min(100, v.ethDepth + (Math.random() - 0.5) * 10)),
          vpin: v.name === "Kraken" ? Math.max(0.6, v.vpin + (Math.random() - 0.5) * 0.1) : Math.max(0.1, Math.min(0.8, v.vpin + (Math.random() - 0.5) * 0.1))
        }))
      }));
    }, 2500);
    return () => clearInterval(timer);
  }, []);

  const getToxicityColor = (vpin: number) => {
    if (vpin > 0.8) return "bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]";
    if (vpin > 0.5) return "bg-orange-400";
    return "bg-emerald-500";
  };

  return (
    <div className="grid gap-6 md:grid-cols-2 mt-8">
      {/* Rebate vs Fee Histogram Card */}
      <div className="rounded-2xl border border-border/80 bg-background/70 p-6 hover:shadow-lg transition-all">
        <div className="flex items-center gap-3 mb-6">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
            <Activity className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">Execution Alpha</h2>
            <p className="text-xs text-muted-foreground">Rebates Collected vs Fees Paid</p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex justify-between items-end mb-2">
            <div>
              <p className="text-2xl font-bold text-positive">+${data.rebates.toFixed(2)}</p>
              <p className="text-xs text-muted-foreground">Maker Rebates</p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-negative">-${data.fees.toFixed(2)}</p>
              <p className="text-xs text-muted-foreground">Taker Fees</p>
            </div>
          </div>
          
          {/* Simple dual bar chart */}
          <div className="relative h-4 w-full bg-muted rounded-full overflow-hidden flex">
            <div className="h-full bg-positive" style={{ width: `${(data.rebates / (data.rebates + data.fees)) * 100}%` }} />
            <div className="h-full bg-negative" style={{ width: `${(data.fees / (data.rebates + data.fees)) * 100}%` }} />
          </div>

          <div className="pt-4 border-t border-border mt-4 text-center">
            <p className="text-sm font-semibold text-foreground">Net Execution Alpha</p>
            <p className="text-xl font-bold text-primary">+${data.netAlpha.toFixed(2)}</p>
          </div>
        </div>
      </div>

      {/* Multi-Venue SOR Depth Map & vPIN Toxicity */}
      <div className="rounded-2xl border border-border/80 bg-background/70 p-6 hover:shadow-lg transition-all">
        <div className="flex items-center gap-3 mb-6">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
            <LayoutGrid className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">Multi-Venue SOR Map</h2>
            <p className="text-xs text-muted-foreground">Liquidity Depth & Flow Toxicity</p>
          </div>
        </div>

        <div className="space-y-5">
          {data.venues.map((venue) => (
            <div key={venue.name} className="flex flex-col gap-2">
              <div className="flex justify-between items-center">
                <span className="font-semibold text-sm text-foreground flex items-center gap-2">
                  <Target className="h-4 w-4 text-muted-foreground" />
                  {venue.name}
                </span>
                
                {venue.vpin > 0.8 && (
                  <span className="text-xs px-2 py-0.5 rounded border border-red-500/50 bg-red-500/10 text-red-500 flex items-center gap-1 animate-pulse">
                    <ShieldAlert className="h-3 w-3" /> Toxic Flow
                  </span>
                )}
              </div>
              
              {/* Depth Bars */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-muted-foreground">
                    <span>BTC/USD Depth</span>
                    <span>{venue.btcDepth.toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-orange-400" style={{ width: `${venue.btcDepth}%` }} />
                  </div>
                </div>
                
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-muted-foreground">
                    <span>ETH/USD Depth</span>
                    <span>{venue.ethDepth.toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-blue-400" style={{ width: `${venue.ethDepth}%` }} />
                  </div>
                </div>
              </div>

              {/* Toxicity Heatmap Bar */}
              <div className="h-1 w-full bg-muted rounded-full overflow-hidden mt-1 flex">
                 <div className={`h-full transition-all duration-500 ${getToxicityColor(venue.vpin)}`} style={{ width: `${venue.vpin * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
