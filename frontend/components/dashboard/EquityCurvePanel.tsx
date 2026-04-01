"use client";

import { useEffect, useRef, useState } from "react";
import { useAuthContext } from "@/lib/auth-context";

interface CurvePoint {
  t: string;
  v: number;
}

interface DdPoint {
  t: string;
  dd: number;
}

interface EquityCurveData {
  curve: CurvePoint[];
  drawdown: DdPoint[];
  peak: number;
  current: number;
  drawdown_pct: number;
  total_points: number;
  note?: string | null;
}

function Sparkline({
  data,
  color,
  height = 60,
}: {
  data: number[];
  color: string;
  height?: number;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  if (!data || data.length < 2) {
    return <div style={{ height }} className="flex items-center justify-center text-xs text-muted-foreground">No data</div>;
  }
  const w = 600;
  const h = height;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 4) - 2;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      className="w-full"
      style={{ height }}
    >
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

export default function EquityCurvePanel() {
  const { token } = useAuthContext();
  const [data, setData] = useState<EquityCurveData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState("");

  const fetchData = async () => {
    try {
      const res = await fetch("/api/v1/equity-curve?points=200", {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        cache: "no-store",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setLastUpdated(new Date().toLocaleTimeString());
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "fetch error");
    }
  };

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 15_000);
    return () => clearInterval(id);
  }, [token]);

  if (error)
    return (
      <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
        Equity Curve error: {error}
      </div>
    );

  if (!data)
    return (
      <div className="p-6 text-sm text-muted-foreground animate-pulse">
        Loading equity curve…
      </div>
    );

  const equityValues = data.curve.map((p) => p.v);
  const ddValues = data.drawdown.map((p) => p.dd);

  const ddColor =
    data.drawdown_pct < -5
      ? "#f87171"
      : data.drawdown_pct < -2
      ? "#facc15"
      : "#4ade80";

  const equityColor =
    data.current >= data.peak * 0.98 ? "#4ade80" : "#facc15";

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-foreground">Equity Curve</h2>
        <span className="text-xs text-muted-foreground">
          Updated {lastUpdated} · auto 15s
          {data.total_points > 0 && (
            <span className="ml-2">({data.total_points.toLocaleString()} pts total)</span>
          )}
        </span>
      </div>

      {data.curve.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {data.note ?? "No equity data yet — trading populates this."}
        </p>
      ) : (
        <>
          {/* Stats row */}
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
              <div className="text-xs text-muted-foreground">Current Equity</div>
              <div className="text-lg font-bold font-mono">${data.current.toLocaleString()}</div>
            </div>
            <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
              <div className="text-xs text-muted-foreground">Peak Equity</div>
              <div className="text-lg font-bold font-mono text-blue-400">
                ${data.peak.toLocaleString()}
              </div>
            </div>
            <div className="rounded-2xl border border-border/60 bg-background/60 p-3">
              <div className="text-xs text-muted-foreground">Current Drawdown</div>
              <div
                className="text-lg font-bold font-mono"
                style={{ color: ddColor }}
              >
                {data.drawdown_pct.toFixed(2)}%
              </div>
            </div>
          </div>

          {/* Equity sparkline */}
          <div className="rounded-2xl border border-border/60 bg-background/60 p-4 space-y-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                Equity Value
              </span>
              <span className="text-xs font-mono" style={{ color: equityColor }}>
                ${data.current.toLocaleString()}
              </span>
            </div>
            <Sparkline data={equityValues} color={equityColor} height={80} />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{data.curve[0]?.t ? new Date(data.curve[0].t).toLocaleDateString() : "—"}</span>
              <span>{data.curve.at(-1)?.t ? new Date(data.curve.at(-1)!.t).toLocaleDateString() : "—"}</span>
            </div>
          </div>

          {/* Drawdown sparkline */}
          {ddValues.length > 1 && (
            <div className="rounded-2xl border border-border/60 bg-background/60 p-4 space-y-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Drawdown %
                </span>
                <span className="text-xs font-mono" style={{ color: ddColor }}>
                  {data.drawdown_pct.toFixed(2)}%
                </span>
              </div>
              <Sparkline data={ddValues} color={ddColor} height={50} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
