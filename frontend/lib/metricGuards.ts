/**
 * lib/metricGuards.ts - Sanitization guards for financial metrics
 *
 * Ensures API values are finite, bounded, and safe for display.
 */

const MONEY_ABS_MAX = 1_000_000_000_000;
const SHARPE_ABS_MAX = 20;
const COUNT_MAX = 1_000_000;

function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function sanitizeMoney(value: unknown, fallback = 0): number {
  if (!isFiniteNumber(value)) return fallback;
  if (Math.abs(value) > MONEY_ABS_MAX) return fallback;
  return value;
}

export function sanitizeCount(value: unknown, fallback = 0): number {
  if (!isFiniteNumber(value)) return fallback;
  const rounded = Math.trunc(value);
  if (rounded < 0 || rounded > COUNT_MAX) return fallback;
  return rounded;
}

export function sanitizeExecutionMetrics(raw: Record<string, unknown>): Record<string, unknown> {
  const capital = sanitizeMoney(raw.capital);
  const startingCapital = sanitizeMoney(raw.starting_capital);
  const dailyPnl = sanitizeMoney(raw.daily_pnl);
  const totalPnl = sanitizeMoney(raw.total_pnl);

  let drawdown = 0;
  const ddRaw = raw.max_drawdown;
  if (isFiniteNumber(ddRaw)) {
    if (Math.abs(ddRaw) > 1 && Math.abs(ddRaw) <= 100) {
      drawdown = ddRaw / 100;
    } else if (Math.abs(ddRaw) <= 1) {
      drawdown = ddRaw;
    }
  }

  let sharpe = 0;
  if (isFiniteNumber(raw.sharpe_ratio) && Math.abs(raw.sharpe_ratio) <= SHARPE_ABS_MAX) {
    sharpe = raw.sharpe_ratio;
  }

  let winRate = 0;
  const wrRaw = raw.win_rate;
  if (isFiniteNumber(wrRaw)) {
    if (wrRaw >= 0 && wrRaw <= 1) winRate = wrRaw;
    else if (wrRaw > 1 && wrRaw <= 100) winRate = wrRaw / 100;
  }

  return {
    ...raw,
    capital,
    starting_capital: startingCapital,
    daily_pnl: dailyPnl,
    total_pnl: totalPnl,
    max_drawdown: drawdown,
    sharpe_ratio: sharpe,
    win_rate: winRate,
    open_positions: sanitizeCount(raw.open_positions),
    total_trades: sanitizeCount(raw.total_trades ?? raw.trades_count),
  };
}
