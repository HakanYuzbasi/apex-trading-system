/**
 * lib/formatters.ts - Display formatting utilities for financial data
 */

export function formatCurrency(value: number): string {
  if (!Number.isFinite(value)) return "$0";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatCurrencyWithCents(value: number): string {
  if (!Number.isFinite(value)) return "$0.00";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatCompactCurrency(value: number): string {
  if (!Number.isFinite(value)) return "$0";
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}K`;
  return formatCurrency(value);
}

export function formatPct(value: number): string {
  if (!Number.isFinite(value)) return "0.0%";
  return `${(value * 100).toFixed(1)}%`;
}

export function normalizeDrawdownPct(value: number | null): number {
  if (value === null) return 0;
  // Convert to 0-100 scale if in 0-1 scale
  const abs = Math.abs(value);
  if (abs <= 1) return abs * 100;
  return abs;
}

export function formatTechnicalQty(value: number | string | null | undefined): string {
  if (value == null) return "—";
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  if (num === 0) return "0";
  
  // For very small numbers, use more decimals, for large numbers use 2-4
  const abs = Math.abs(num);
  if (abs < 0.0001) return num.toExponential(4);
  
  // Format with up to 4 decimals, but strip trailing zeros
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 4,
    minimumFractionDigits: 0,
    useGrouping: true,
  }).format(num);
}

export function formatMetric(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatSleeveLabel(sleeve: string): string {
  return sleeve
    .replace(/_sleeve$/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function sortIndicator(active: boolean, direction: "asc" | "desc"): string {
  if (!active) return "";
  return direction === "asc" ? " \u25B2" : " \u25BC";
}

export function clampPct(value: number | null | undefined, min = 0, max = 100): number {
  if (value == null || !Number.isFinite(Number(value))) return min;
  return Math.min(max, Math.max(min, Number(value)));
}
