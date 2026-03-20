"use client";

import { useMemo, useState } from "react";
import { Activity } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

export type HeatmapPosition = {
  symbol: string;
  signal?: number;
  /** Raw composite signal value, -1 to +1 */
  composite_signal?: number;
  /** Direction string from WebSocket */
  signal_direction?: string;
  /** Whether a live open position exists for this symbol */
  has_position?: boolean;
  /** ISO timestamp of last update */
  last_updated?: string;
};

export type SignalHeatmapProps = {
  /** Positions dict from WebSocket state (symbol -> position data) */
  positions?: Record<string, unknown> | null;
  /** Optional pre-built list (takes precedence over raw `positions` dict) */
  signalData?: HeatmapPosition[];
};

// ── Columns ───────────────────────────────────────────────────────────────────

const COLUMNS = ["ML", "Technical", "Sentiment", "Momentum", "Composite"] as const;
type Column = (typeof COLUMNS)[number];

// ── Color helpers ─────────────────────────────────────────────────────────────

function cellBg(value: number | null): string {
  if (value === null) return "#e5e7eb"; // N/A — neutral grey
  if (value >= 0.15) return "#16a34a";  // bright green
  if (value >= 0.05) return "#86efac";  // light green
  if (value > -0.05) return "#e5e7eb";  // neutral grey
  if (value > -0.15) return "#fca5a5";  // light red
  return "#dc2626";                      // bright red
}

function cellTextColor(value: number | null): string {
  if (value === null) return "#6b7280";       // muted for N/A
  if (value >= 0.15) return "#ffffff";        // white on bright green
  if (value >= 0.05) return "#14532d";        // dark green on light green
  if (value > -0.05) return "#6b7280";        // muted on neutral
  if (value > -0.15) return "#7f1d1d";        // dark red on light red
  return "#ffffff";                            // white on bright red
}

// ── Relative time formatter ───────────────────────────────────────────────────

function relativeTime(isoTs: string | undefined): string {
  if (!isoTs) return "";
  try {
    const diff = Math.floor((Date.now() - new Date(isoTs).getTime()) / 1000);
    if (diff < 0) return "just now";
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  } catch {
    return "";
  }
}

// ── Value display ─────────────────────────────────────────────────────────────

function fmtVal(v: number | null): string {
  if (v === null) return "N/A";
  return (v >= 0 ? "+" : "") + v.toFixed(2);
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function SignalHeatmap({ positions, signalData }: SignalHeatmapProps) {
  const [lastRefresh] = useState<Date>(new Date());

  // Build the row data from whichever source is provided
  const rows: HeatmapPosition[] = useMemo(() => {
    // If caller provided pre-built data, use that directly
    if (signalData && signalData.length > 0) {
      return signalData
        .slice()
        .sort(
          (a, b) =>
            Math.abs(b.composite_signal ?? b.signal ?? 0) -
            Math.abs(a.composite_signal ?? a.signal ?? 0),
        )
        .slice(0, 20);
    }

    // Otherwise derive from the WebSocket positions dict
    if (!positions || typeof positions !== "object") return [];

    const entries = Object.entries(positions)
      .map(([symbol, raw]) => {
        const data =
          raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};

        const composite =
          data.composite_signal !== undefined
            ? Number(data.composite_signal)
            : data.current_signal !== undefined
            ? Number(data.current_signal)
            : data.signal !== undefined
            ? Number(data.signal)
            : null;

        const qty = Number(data.qty ?? 0);

        return {
          symbol: String(symbol).trim().toUpperCase(),
          composite_signal: composite !== null && Number.isFinite(composite) ? composite : undefined,
          signal_direction: String(data.signal_direction ?? ""),
          has_position: Number.isFinite(qty) && qty !== 0,
          last_updated: data.last_updated
            ? String(data.last_updated)
            : data.timestamp
            ? String(data.timestamp)
            : undefined,
        } satisfies HeatmapPosition;
      })
      .filter((r) => r.composite_signal !== undefined || r.has_position);

    // Sort descending by absolute composite signal strength
    entries.sort(
      (a, b) =>
        Math.abs(b.composite_signal ?? 0) - Math.abs(a.composite_signal ?? 0),
    );

    return entries.slice(0, 20);
  }, [positions, signalData]);

  // Resolve cell value for a given symbol + column
  // Only the Composite column has real data from the WS feed.
  // All other columns will show N/A until the backend exposes them.
  function cellValue(row: HeatmapPosition, col: Column): number | null {
    if (col === "Composite") {
      return row.composite_signal !== undefined ? row.composite_signal : null;
    }
    // ML / Technical / Sentiment / Momentum — not yet in the WS payload
    return null;
  }

  const refreshLabel = relativeTime(lastRefresh.toISOString());

  return (
    <section className="hidden md:block apex-fade-up">
      <article className="apex-panel rounded-2xl p-5">
        {/* Header */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            <div>
              <h2 className="text-sm font-semibold text-foreground">Signal Heatmap</h2>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                Top {rows.length} symbols by absolute composite signal strength
              </p>
            </div>
          </div>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-secondary px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-secondary-foreground">
            <span
              className="inline-block h-1.5 w-1.5 rounded-full bg-positive animate-pulse"
              aria-hidden="true"
            />
            {refreshLabel ? `Refreshed ${refreshLabel}` : "Live"}
          </span>
        </div>

        {/* Legend */}
        <div className="mb-3 flex flex-wrap items-center gap-3 text-[10px] font-medium text-muted-foreground">
          <span className="font-semibold uppercase tracking-wide">Scale:</span>
          {(
            [
              ["#16a34a", "#fff", "Strong Long (≥0.15)"],
              ["#86efac", "#14532d", "Long (≥0.05)"],
              ["#e5e7eb", "#6b7280", "Neutral"],
              ["#fca5a5", "#7f1d1d", "Short (≤−0.05)"],
              ["#dc2626", "#fff", "Strong Short (≤−0.15)"],
            ] as [string, string, string][]
          ).map(([bg, fg, label]) => (
            <span
              key={label}
              className="inline-flex items-center gap-1 rounded px-2 py-0.5"
              style={{ backgroundColor: bg, color: fg }}
            >
              {label}
            </span>
          ))}
        </div>

        {/* Grid */}
        {rows.length === 0 ? (
          <div className="flex h-24 items-center justify-center rounded-xl border border-border/70 bg-background/60">
            <p className="text-xs text-muted-foreground">
              No signal data available. Waiting for WebSocket feed...
            </p>
          </div>
        ) : (
          <div
            className="overflow-auto rounded-xl border border-border/80"
            style={{ maxHeight: "400px" }}
          >
            <table className="min-w-full border-collapse text-xs">
              <thead className="sticky top-0 z-10 bg-background/95 backdrop-blur">
                <tr>
                  {/* Symbol column header */}
                  <th className="px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-muted-foreground whitespace-nowrap border-b border-border/80">
                    Symbol
                  </th>
                  {/* Signal component column headers */}
                  {COLUMNS.map((col) => (
                    <th
                      key={col}
                      className="px-3 py-2 text-center text-[11px] font-semibold uppercase tracking-wide text-muted-foreground whitespace-nowrap border-b border-border/80"
                    >
                      {col}
                    </th>
                  ))}
                  {/* Last update column */}
                  <th className="px-3 py-2 text-right text-[11px] font-semibold uppercase tracking-wide text-muted-foreground whitespace-nowrap border-b border-border/80">
                    Updated
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, idx) => (
                  <tr
                    key={row.symbol}
                    className={`border-b border-border/60 hover:bg-secondary/30 transition-colors ${
                      idx % 2 === 0 ? "bg-background/40" : "bg-background/20"
                    }`}
                  >
                    {/* Symbol cell */}
                    <td className="px-3 py-2 whitespace-nowrap">
                      <span
                        className={`font-mono text-[11px] ${
                          row.has_position
                            ? "font-bold text-foreground"
                            : "font-medium text-muted-foreground"
                        }`}
                        title={row.has_position ? "Open position" : "No open position"}
                      >
                        {row.symbol}
                      </span>
                      {row.has_position && (
                        <span
                          className="ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-positive"
                          aria-label="Open position"
                        />
                      )}
                    </td>

                    {/* Signal component cells */}
                    {COLUMNS.map((col) => {
                      const val = cellValue(row, col);
                      const bg = cellBg(val);
                      const fg = cellTextColor(val);
                      return (
                        <td key={col} className="px-1.5 py-1.5 text-center">
                          <span
                            className="inline-block min-w-[56px] rounded px-2 py-1 font-mono text-[11px] font-semibold tabular-nums"
                            style={{ backgroundColor: bg, color: fg }}
                          >
                            {fmtVal(val)}
                          </span>
                        </td>
                      );
                    })}

                    {/* Last updated */}
                    <td className="px-3 py-2 text-right font-mono text-[10px] text-muted-foreground whitespace-nowrap">
                      {relativeTime(row.last_updated)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </article>
    </section>
  );
}
