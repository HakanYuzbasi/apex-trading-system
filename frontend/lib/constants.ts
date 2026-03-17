/**
 * lib/constants.ts - Dashboard constants and targets
 */

export const MAX_POSITIONS = 40;
export const DRAWDOWN_BUDGET_PCT = 10; // 10% max drawdown
export const SHARPE_TARGET = 1.5; // Updated from 0.8 to 1.5 for split-session mode
export const WIN_RATE_TARGET = 0.57;
export const RETURN_CYCLE_TARGET = 0.05; // 5% per cycle
export const TRADE_CYCLE_TARGET = 200;
export const EDGE_CAPTURE_TARGET = 0.6; // 60% edge capture

// Session types
export const SESSION_TYPES = ["core", "crypto"] as const;
export type SessionType = (typeof SESSION_TYPES)[number];

// Per-session targets
export const SESSION_CONFIG = {
  core: {
    label: "Core Strategy",
    description: "Equities, indices, and forex",
    sharpeTarget: 1.5,
    maxPositions: 25,
    initialCapital: 1_000_000,
  },
  crypto: {
    label: "Crypto Sleeve",
    description: "Cryptocurrency trading",
    sharpeTarget: 1.5,
    maxPositions: 15,
    initialCapital: 100_000,
  },
} as const;
