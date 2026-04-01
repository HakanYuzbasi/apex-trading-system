/**
 * @jest-environment node
 */
// tests for /api/v1/mission-control route — fetch mocked, no network calls
import { jest } from "@jest/globals";
import { NextRequest } from "next/server";

const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

const makeReq = (token?: string) =>
  new NextRequest("http://localhost/api/v1/mission-control", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

// ── helpers ───────────────────────────────────────────────────────────────────

function mockBackend(
  stateMock: object | null,
  regimeMock: object | null = null,
  rlMock: object | null = null,
  univMock: object | null = null,
) {
  let call = 0;
  mockFetch.mockImplementation(() => {
    const responses = [stateMock, regimeMock, rlMock, univMock];
    const payload = responses[call++];
    if (payload === null)
      return Promise.resolve({ ok: false, status: 503, json: async () => null } as Response);
    return Promise.resolve({
      ok: true,
      status: 200,
      json: async () => payload,
    } as Response);
  });
}

// ── tests ─────────────────────────────────────────────────────────────────────

describe("GET /api/v1/mission-control", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with required top-level keys", async () => {
    mockBackend(
      { equity: 100_000, daily_pnl: 500, positions: {}, max_positions: 40, regime: "bull", vix: 18, kill_switch_active: false, governor_tier: "GREEN", position_details: [] },
      { prediction: { probability: 0.30, direction: "unknown", size_multiplier: 1.0 } },
      { epsilon: 0.12, total_updates: 250 },
      { report: { scored_count: 80 } },
    );
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("system");
    expect(body).toHaveProperty("risk_budget");
    expect(body).toHaveProperty("top_positions");
    expect(body).toHaveProperty("predictive");
    expect(body).toHaveProperty("timestamp");
  });

  it("system block reflects state fields", async () => {
    mockBackend(
      { equity: 500_000, daily_pnl: -2000, positions: {}, max_positions: 40, regime: "bear", vix: 32, kill_switch_active: true, governor_tier: "RED", position_details: [] },
      null, null, null,
    );
    const body = await (await GET(makeReq())).json();
    expect(body.system.regime).toBe("bear");
    expect(body.system.vix).toBe(32);
    expect(body.system.kill_switch_active).toBe(true);
    expect(body.system.governor_tier).toBe("RED");
  });

  it("risk_budget daily_pnl_pct computed correctly", async () => {
    mockBackend(
      { equity: 100_000, daily_pnl: 1000, positions: {}, max_positions: 40, regime: "neutral", vix: 18, kill_switch_active: false, governor_tier: "GREEN", position_details: [] },
      null, null, null,
    );
    const body = await (await GET(makeReq())).json();
    expect(body.risk_budget.daily_pnl_pct).toBeCloseTo(1.0, 1);
  });

  it("positions_pct computed correctly", async () => {
    mockBackend(
      { equity: 100_000, daily_pnl: 0, positions: { AAPL: 10, MSFT: 5, TSLA: 8 }, max_positions: 40, regime: "neutral", vix: 18, kill_switch_active: false, governor_tier: "GREEN", position_details: [] },
      null, null, null,
    );
    const body = await (await GET(makeReq())).json();
    expect(body.risk_budget.position_count).toBe(3);
    expect(body.risk_budget.positions_pct).toBe(8); // 3/40 = 7.5 → 8 rounded
  });

  it("top_positions sorted by abs pnl_pct", async () => {
    const details = [
      { symbol: "AAPL", pnl_pct: 0.01, pnl: 100, qty: 10, side: "LONG", signal_direction: "bullish" },
      { symbol: "GME",  pnl_pct: -0.08, pnl: -800, qty: 5, side: "LONG", signal_direction: "bearish" },
      { symbol: "TSLA", pnl_pct: 0.05, pnl: 500, qty: 8, side: "LONG", signal_direction: "bullish" },
    ];
    mockBackend(
      { equity: 100_000, daily_pnl: 0, positions: {}, max_positions: 40, regime: "neutral", vix: 18, kill_switch_active: false, governor_tier: "GREEN", position_details: details },
      null, null, null,
    );
    const body = await (await GET(makeReq())).json();
    expect(body.top_positions[0].symbol).toBe("GME");   // |-8%| highest
    expect(body.top_positions[1].symbol).toBe("TSLA");  // |+5%|
    expect(body.top_positions[2].symbol).toBe("AAPL");  // |+1%|
  });

  it("predictive block includes regime transition data", async () => {
    mockBackend(
      { equity: 100_000, daily_pnl: 0, positions: {}, max_positions: 40, regime: "neutral", vix: 18, kill_switch_active: false, governor_tier: "GREEN", position_details: [] },
      { prediction: { probability: 0.72, direction: "risk_off", size_multiplier: 0.78 } },
      { epsilon: 0.05, total_updates: 1000 },
      { report: { scored_count: 60 } },
    );
    const body = await (await GET(makeReq())).json();
    expect(body.predictive.transition_probability).toBeCloseTo(0.72);
    expect(body.predictive.transition_direction).toBe("risk_off");
    expect(body.predictive.rl_total_updates).toBe(1000);
    expect(body.predictive.universe_scored).toBe(60);
  });

  it("handles backend failures gracefully — no crash", async () => {
    // All downstream calls fail
    mockFetch.mockRejectedValue(new Error("network error"));
    const body = await (await GET(makeReq())).json();
    expect(body).toHaveProperty("system");
    expect(body.system.equity).toBe(0);
    expect(body.predictive.transition_probability).toBeNull();
  });

  it("top_positions capped at 5", async () => {
    const details = Array.from({ length: 10 }, (_, i) => ({
      symbol: `SYM${i}`, pnl_pct: i * 0.01, pnl: i * 100, qty: 10, side: "LONG", signal_direction: "bullish",
    }));
    mockBackend(
      { equity: 100_000, daily_pnl: 0, positions: {}, max_positions: 40, regime: "neutral", vix: 18, kill_switch_active: false, governor_tier: "GREEN", position_details: details },
      null, null, null,
    );
    const body = await (await GET(makeReq())).json();
    expect(body.top_positions.length).toBeLessThanOrEqual(5);
  });

  it("timestamp is ISO string", async () => {
    mockBackend(
      { equity: 0, daily_pnl: 0, positions: {}, max_positions: 40, regime: "neutral", vix: 0, kill_switch_active: false, governor_tier: "GREEN", position_details: [] },
      null, null, null,
    );
    const body = await (await GET(makeReq())).json();
    expect(() => new Date(body.timestamp)).not.toThrow();
  });
});
