const MONEY_ABS_MAX = 1_000_000_000_000;
const SHARPE_ABS_MAX = 200; // raised from 20 — cold-start bootstrapped Sharpe can exceed 50
const COUNT_MAX = 1_000_000;

export type SessionType = "core" | "crypto" | "unified";

type SessionEnvelope = {
  session_type: SessionType;
  available: boolean;
  error: string | null;
  upstream_status: number | null;
  timestamp: string | null;
};

export type SessionStatusPayload = SessionEnvelope & {
  status: string | null;
  initial_capital: number | null;
  symbols_count: number | null;
  capital: number | null;
  starting_capital: number | null;
  daily_pnl: number | null;
  daily_pnl_realized: number | null;
  total_pnl: number | null;
  max_drawdown: number | null;
  sharpe_ratio: number | null;
  win_rate: number | null;
  open_positions: number;
  total_trades: number;
};

export type SessionMetricsPayload = SessionEnvelope & {
  initial_capital: number | null;
  capital: number | null;
  starting_capital: number | null;
  daily_pnl: number | null;
  daily_pnl_realized: number | null;
  total_pnl: number | null;
  max_drawdown: number | null;
  sharpe_ratio: number | null;
  win_rate: number | null;
  open_positions: number;
  option_positions: number;
  open_positions_total: number;
  total_trades: number;
  sharpe_target: number | null;
  max_positions: number | null;
  signal_threshold: number | null;
  confidence_threshold: number | null;
};

export type SessionPositionRow = {
  symbol: string;
  qty: number;
  side: string;
  entry: number | null;
  current: number | null;
  pnl: number | null;
  pnl_pct: number | null;
  signal: number | null;
  signal_direction: string;
};

export type SessionPositionsPayload = SessionEnvelope & {
  positions: SessionPositionRow[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function asString(value: unknown): string | null {
  const parsed = String(value ?? "").trim();
  return parsed.length > 0 ? parsed : null;
}

function asFiniteNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function asMoney(value: unknown): number | null {
  const parsed = asFiniteNumber(value);
  if (parsed === null || Math.abs(parsed) > MONEY_ABS_MAX) {
    return null;
  }
  return parsed;
}

function asSharpe(value: unknown): number | null {
  const parsed = asFiniteNumber(value);
  if (parsed === null || Math.abs(parsed) > SHARPE_ABS_MAX) {
    return null;
  }
  return parsed;
}

function asCount(value: unknown): number {
  const parsed = asFiniteNumber(value);
  if (parsed === null) {
    return 0;
  }

  const rounded = Math.trunc(parsed);
  if (rounded < 0 || rounded > COUNT_MAX) {
    return 0;
  }

  return rounded;
}

function asDrawdown(value: unknown): number | null {
  const parsed = asFiniteNumber(value);
  if (parsed === null) {
    return null;
  }

  if (Math.abs(parsed) > 100) {
    return null;
  }

  if (Math.abs(parsed) > 1) {
    return parsed / 100;
  }

  return parsed;
}

function asWinRate(value: unknown): number | null {
  const parsed = asFiniteNumber(value);
  if (parsed === null) {
    return null;
  }

  if (parsed >= 0 && parsed <= 1) {
    return parsed;
  }

  if (parsed > 1 && parsed <= 100) {
    return parsed / 100;
  }

  return null;
}

function baseEnvelope(
  sessionType: SessionType,
  payload: Record<string, unknown>,
  error: string | null,
  upstreamStatus: number | null,
): SessionEnvelope {
  const hasData = Object.keys(payload).length > 0;

  return {
    session_type: sessionType,
    available: hasData && !error,
    error,
    upstream_status: upstreamStatus,
    timestamp: asString(payload.timestamp),
  };
}

export function isValidSessionType(value: string): value is SessionType {
  return value === "core" || value === "crypto" || value === "unified";
}

export function sanitizeSessionStatusPayload(
  sessionType: SessionType,
  input: unknown,
  error: string | null = null,
  upstreamStatus: number | null = null,
): SessionStatusPayload {
  const payload = isRecord(input) ? input : {};
  const envelope = baseEnvelope(sessionType, payload, error, upstreamStatus);

  return {
    ...envelope,
    status: asString(payload.status),
    initial_capital: asMoney(payload.initial_capital),
    symbols_count: asFiniteNumber(payload.symbols_count),
    capital: asMoney(payload.capital),
    starting_capital: asMoney(payload.starting_capital ?? payload.initial_capital),
    daily_pnl: asMoney(payload.daily_pnl),
    daily_pnl_realized: asMoney(payload.daily_pnl_realized ?? payload.realized_pnl ?? payload.daily_pnl),
    total_pnl: asMoney(payload.total_pnl),
    max_drawdown: asDrawdown(payload.max_drawdown),
    sharpe_ratio: asSharpe(payload.sharpe_ratio),
    win_rate: asWinRate(payload.win_rate),
    open_positions: asCount(payload.open_positions),
    total_trades: asCount(payload.total_trades ?? payload.trades_count),
  };
}

export function sanitizeSessionMetricsPayload(
  sessionType: SessionType,
  input: unknown,
  error: string | null = null,
  upstreamStatus: number | null = null,
): SessionMetricsPayload {
  const payload = isRecord(input) ? input : {};
  const envelope = baseEnvelope(sessionType, payload, error, upstreamStatus);

  return {
    ...envelope,
    initial_capital: asMoney(payload.initial_capital),
    capital: asMoney(payload.capital),
    starting_capital: asMoney(payload.starting_capital ?? payload.initial_capital),
    daily_pnl: asMoney(payload.daily_pnl),
    daily_pnl_realized: asMoney(payload.daily_pnl_realized ?? payload.realized_pnl ?? payload.daily_pnl),
    total_pnl: asMoney(payload.total_pnl),
    max_drawdown: asDrawdown(payload.max_drawdown),
    sharpe_ratio: asSharpe(payload.sharpe_ratio),
    win_rate: asWinRate(payload.win_rate),
    open_positions: asCount(payload.open_positions),
    option_positions: asCount(payload.option_positions),
    open_positions_total: asCount(payload.open_positions_total ?? payload.open_positions),
    total_trades: asCount(payload.total_trades ?? payload.trades_count),
    sharpe_target: asSharpe(payload.sharpe_target),
    max_positions: asFiniteNumber(payload.max_positions),
    signal_threshold: asFiniteNumber(payload.signal_threshold),
    confidence_threshold: asFiniteNumber(payload.confidence_threshold),
  };
}

export function sanitizeSessionPositionsPayload(
  sessionType: SessionType,
  input: unknown,
  error: string | null = null,
  upstreamStatus: number | null = null,
): SessionPositionsPayload {
  const payload = isRecord(input) ? input : {};
  const rows = Array.isArray(payload.positions) ? payload.positions : [];

  const positions = rows
    .map((row) => {
      if (!isRecord(row)) {
        return null;
      }

      const symbol = asString(row.symbol);
      if (!symbol) {
        return null;
      }

      return {
        symbol,
        qty: asFiniteNumber(row.qty) ?? 0,
        side: asString(row.side) ?? (Number(row.qty) < 0 ? "SHORT" : "LONG"),
        entry: asMoney(row.entry ?? row.avg_price),
        current: asMoney(row.current ?? row.current_price),
        pnl: asMoney(row.pnl),
        pnl_pct: asFiniteNumber(row.pnl_pct),
        signal: asFiniteNumber(row.signal ?? row.current_signal),
        signal_direction: asString(row.signal_direction) ?? "UNKNOWN",
      } satisfies SessionPositionRow;
    })
    .filter((row): row is SessionPositionRow => row !== null);

  return {
    ...baseEnvelope(sessionType, payload, error, upstreamStatus),
    positions,
  };
}
