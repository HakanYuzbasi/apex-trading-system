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

export function clampPct(value: number, min = 0, max = 100): number {
  return Math.min(max, Math.max(min, value));
}

export function normalizeDrawdownPct(value: number): number {
  // Convert to 0-100 scale if in 0-1 scale
  if (Math.abs(value) <= 1) return Math.abs(value) * 100;
  return Math.abs(value);
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
