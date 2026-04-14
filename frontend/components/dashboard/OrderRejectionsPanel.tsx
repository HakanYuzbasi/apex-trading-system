"use client";

import { useEffect, useState } from "react";
import { useAuthContext } from "@/components/auth/AuthProvider";

interface RejectionRecord {
  event_id: string;
  timestamp: string;
  symbol: string;
  asset_class: string;
  side: string;
  quantity: number;
  price: number;
  reason_code: string;
  message: string;
  metadata: Record<string, number | string | boolean>;
  actor: string;
}

interface RejectionsData {
  available: boolean;
  note?: string;
  total_scanned?: number;
  total_rejected?: number;
  reason_breakdown?: Record<string, number>;
  rejections?: RejectionRecord[];
}

const REASON_LABELS: Record<string, string> = {
  max_order_notional: "Notional Cap",
  price_band: "Price Band",
  adv_participation: "ADV Cap",
  gross_exposure: "Gross Exposure",
  invalid_order: "Invalid Order",
  disabled: "Gateway Off",
};

const REASON_COLORS: Record<string, string> = {
  max_order_notional: "text-orange-400 bg-orange-400/10 border-orange-400/30",
  price_band: "text-yellow-400 bg-yellow-400/10 border-yellow-400/30",
  adv_participation: "text-blue-400 bg-blue-400/10 border-blue-400/30",
  gross_exposure: "text-red-400 bg-red-400/10 border-red-400/30",
  invalid_order: "text-red-500 bg-red-500/10 border-red-500/30",
};

function reasonBadge(code: string) {
  const label = REASON_LABELS[code] ?? code.replace(/_/g, " ");
  const color = REASON_COLORS[code] ?? "text-muted-foreground bg-secondary/40 border-border/50";
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${color}`}>
      {label}
    </span>
  );
}

function fmtTime(iso: string): string {
  if (!iso) return "—";
  try {
    return iso.slice(11, 19) + " UTC";
  } catch {
    return iso;
  }
}

function fmtPrice(p: number): string {
  if (p == null || !isFinite(p)) return "—";
  return `$${p.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export default function OrderRejectionsPanel() {
  const { accessToken: token } = useAuthContext();
  const [data, setData] = useState<RejectionsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const params = new URLSearchParams({ limit: "100" });
        const res = await fetch(`/api/v1/order-rejections?${params}`, {
          headers: token ? { authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const id = setInterval(load, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [token]);

  if (loading) return <div className="p-6 text-muted-foreground text-sm">Loading order rejections…</div>;
  if (error) return <div className="p-6 text-red-400 text-sm">Error: {error}</div>;
  if (!data || !data.available) {
    return (
      <div className="p-6 text-muted-foreground text-sm">
        {data?.note ?? "Order rejections unavailable — engine may not be running."}
      </div>
    );
  }

  const rejBreakdown = Object.entries(data.reason_breakdown ?? {}).sort((a, b) => b[1] - a[1]);
  const rows = (data.rejections ?? []).filter(
    (r) => !filter || r.reason_code === filter
  );

  return (
    <div className="p-4 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-foreground">Order Rejections</h2>
        <span className="text-[11px] text-muted-foreground font-mono">
          {data.total_scanned ?? 0} scanned · {data.total_rejected ?? 0} blocked
        </span>
      </div>

      {/* Summary badges */}
      {rejBreakdown.length > 0 && (
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setFilter("")}
            className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors
              ${filter === "" ? "border-primary/60 bg-primary/10 text-primary" : "border-border/50 bg-background/50 text-muted-foreground hover:border-border"}`}
          >
            All
            <span className="font-mono font-semibold">{data.total_rejected ?? 0}</span>
          </button>
          {rejBreakdown.map(([code, count]) => (
            <button
              key={code}
              onClick={() => setFilter(filter === code ? "" : code)}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors
                ${filter === code ? "border-primary/60 bg-primary/10 text-primary" : "border-border/50 bg-background/50 text-muted-foreground hover:border-border"}`}
            >
              {REASON_LABELS[code] ?? code.replace(/_/g, " ")}
              <span className="font-mono font-semibold">{count}</span>
            </button>
          ))}
        </div>
      )}

      {/* No rejections happy path */}
      {rows.length === 0 && (
        <div className="flex items-center justify-center rounded-lg border border-border/40 bg-green-400/5 p-8">
          <p className="text-sm text-green-400">No pre-trade rejections{filter ? ` for "${REASON_LABELS[filter] ?? filter}"` : ""} in the last 48h.</p>
        </div>
      )}

      {/* Rejections table */}
      {rows.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-border/60">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-border/60 text-muted-foreground text-[11px] uppercase">
                <th className="px-3 py-2 text-left">Time</th>
                <th className="px-3 py-2 text-left">Symbol</th>
                <th className="px-3 py-2 text-left">Side</th>
                <th className="px-3 py-2 text-right">Qty</th>
                <th className="px-3 py-2 text-right">Price</th>
                <th className="px-3 py-2 text-left">Reason</th>
                <th className="px-3 py-2 text-left">Detail</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.event_id}
                  className="border-b border-border/40 hover:bg-secondary/30 transition-colors"
                >
                  <td className="px-3 py-1.5 text-muted-foreground whitespace-nowrap">
                    {fmtTime(r.timestamp)}
                  </td>
                  <td className="px-3 py-1.5 text-foreground font-semibold">
                    {r.symbol}
                    {r.asset_class?.toLowerCase() === "crypto" && (
                      <span className="ml-1 text-[9px] text-blue-400 uppercase">crypto</span>
                    )}
                  </td>
                  <td className={`px-3 py-1.5 ${r.side === "BUY" ? "text-green-400" : "text-red-400"}`}>
                    {r.side}
                  </td>
                  <td className="px-3 py-1.5 text-right text-muted-foreground">
                    {r.quantity?.toLocaleString("en-US")}
                  </td>
                  <td className="px-3 py-1.5 text-right text-muted-foreground">
                    {fmtPrice(r.price)}
                  </td>
                  <td className="px-3 py-1.5">
                    {reasonBadge(r.reason_code)}
                  </td>
                  <td className="px-3 py-1.5 text-muted-foreground max-w-[280px] truncate" title={r.message}>
                    {r.message}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
