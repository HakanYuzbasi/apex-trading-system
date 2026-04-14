import { NextRequest, NextResponse } from "next/server";
import { buildAuthHeaders, getBackendApiBase, getRequestToken } from "@/app/api/_lib/backend";

export const dynamic = "force-dynamic";
export const revalidate = 0;

/** Normalise PositionSide enum strings the engine may emit. */
function normalizeSide(raw: unknown): string {
  const s = String(raw ?? "").toUpperCase();
  if (s.includes("SHORT")) return "SHORT";
  return "LONG";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function asNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function asIsoString(value: unknown): string | null {
  const parsed = String(value ?? "").trim();
  return parsed.length > 0 ? parsed : null;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => String(item ?? "").trim())
    .filter(Boolean);
}

function sanitizeBrokerPosition(row: unknown, trackedSymbols: Set<string>) {
  if (!isRecord(row)) {
    return null;
  }

  const symbol = String(row.symbol ?? row.normalized_symbol ?? "").trim().toUpperCase();
  if (!symbol) {
    return null;
  }

  return {
    symbol,
    normalized_symbol: String(row.normalized_symbol ?? symbol).trim().toUpperCase() || symbol,
    qty: asNumber(row.qty) ?? 0,
    side: normalizeSide(row.side),
    market_value: asNumber(row.market_value),
    unrealized_pl: asNumber(row.unrealized_pl),
    unrealized_plpc: asNumber(row.unrealized_plpc),
    current_price: asNumber(row.current_price),
    avg_price: asNumber(row.avg_price),
    is_orphaned: trackedSymbols.size === 0 ? true : !trackedSymbols.has(symbol),
  };
}

export async function GET(request: NextRequest) {
  const token = getRequestToken(request);
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...buildAuthHeaders(token),
  };
  const backendApiBase = getBackendApiBase();

  try {
    const statusRes = await fetch(`${backendApiBase}/status`, {
      headers,
      cache: "no-store",
    });
    const statusData = await statusRes.json().catch(() => null);

    // The trading engine embeds broker_positions and active_pairs in the
    // trading_state.json file but exposes them via the state endpoint.
    // We need to read the state file reader — pull from the state endpoint
    // which proxies through /status.
    // If broker_positions aren't on /status, try the state endpoint.
    let brokerPositions: unknown[] = [];
    let activePairs: string[] = [];
    let timestamp: string | null = null;
    let error: string | null = null;

    if (isRecord(statusData)) {
      if (Array.isArray(statusData.broker_positions)) {
        brokerPositions = statusData.broker_positions;
      }
      activePairs = asStringArray(statusData.active_pairs);
      timestamp = asIsoString(statusData.timestamp);
    } else if (!statusRes.ok) {
      error = `status_${statusRes.status}`;
    }

    // If broker_positions absent from /status, read from state file via /state
    if (brokerPositions.length === 0) {
      try {
        const stateRes = await fetch(`${backendApiBase}/state`, {
          headers,
          cache: "no-store",
        });
        const stateData = await stateRes.json().catch(() => null);
        if (isRecord(stateData)) {
          if (Array.isArray(stateData.broker_positions)) {
            brokerPositions = stateData.broker_positions;
          }
          if (activePairs.length === 0) {
            activePairs = asStringArray(stateData.active_pairs);
          }
          timestamp = timestamp ?? asIsoString(stateData.timestamp);
        } else if (!stateRes.ok && !error) {
          error = `state_${stateRes.status}`;
        }
      } catch (stateError) {
        if (!error) {
          error = stateError instanceof Error ? stateError.message : "state_unreachable";
        }
      }
    }

    // Build a set of symbols that are covered by active APEX strategy pairs
    // active_pairs format: "SYMBOL_A/SYMBOL_B"
    const trackedSymbols = new Set<string>();
    for (const pair of activePairs) {
      const parts = String(pair).split("/");
      parts.forEach((p) => trackedSymbols.add(p.trim().toUpperCase()));
    }

    // Annotate each position with is_orphaned and normalise side
    const annotated = brokerPositions
      .map((position) => sanitizeBrokerPosition(position, trackedSymbols))
      .filter((position): position is NonNullable<typeof position> => position !== null);

    if (annotated.length === 0 && !error && !statusRes.ok) {
      error = `upstream_${statusRes.status}`;
    }

    return NextResponse.json({
      available: annotated.length > 0 || activePairs.length > 0,
      broker_positions: annotated,
      active_pairs: activePairs,
      timestamp,
      total: annotated.length,
      tracked: annotated.filter((position) => !position.is_orphaned).length,
      orphaned: annotated.filter((position) => position.is_orphaned).length,
      error,
    });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      {
        available: false,
        broker_positions: [],
        active_pairs: [],
        timestamp: null,
        total: 0,
        tracked: 0,
        orphaned: 0,
        error: msg,
      },
    );
  }
}
