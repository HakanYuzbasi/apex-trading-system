/**
 * @jest-environment node
 */
import { jest } from "@jest/globals";
import { NextRequest } from "next/server";

const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

function makeReq(url = "http://localhost/api/v1/portfolio-heat"): NextRequest {
  return new NextRequest(url);
}

const SAMPLE_HEAT = {
  positions: [
    { symbol: "AAPL", qty: 10, notional: 1800.0, asset_class: "equity", weight_pct: 60.0 },
    { symbol: "CRYPTO:BTC/USD", qty: 0.05, notional: 1200.0, asset_class: "crypto", weight_pct: 40.0 },
  ],
  by_asset_class: {
    equity: { count: 1, notional: 1800.0, weight_pct: 60.0 },
    crypto: { count: 1, notional: 1200.0, weight_pct: 40.0 },
  },
  total_notional: 3000.0,
  position_count: 2,
  hhi_concentration: 0.52,
  regime: "neutral",
  vix: 22.5,
  alpha_decay: { optimal_hold_hours: 5.0, alpha_half_life: 12.0 },
  model_drift: { health: "healthy", should_retrain: false, ic_current: 0.07, hit_rate_current: 0.58 },
};

describe("GET /api/v1/portfolio-heat", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns portfolio heat data on success", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => SAMPLE_HEAT,
    });
    const res = await GET(makeReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.position_count).toBe(2);
    expect(body.total_notional).toBeCloseTo(3000.0);
    expect(body.hhi_concentration).toBeDefined();
  });

  it("returns positions array", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => SAMPLE_HEAT,
    });
    const res = await GET(makeReq());
    const body = await res.json();
    expect(Array.isArray(body.positions)).toBe(true);
    expect(body.positions).toHaveLength(2);
  });

  it("returns asset class breakdown", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => SAMPLE_HEAT,
    });
    const res = await GET(makeReq());
    const body = await res.json();
    expect(body.by_asset_class.equity).toBeDefined();
    expect(body.by_asset_class.crypto).toBeDefined();
  });

  it("includes model drift status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => SAMPLE_HEAT,
    });
    const res = await GET(makeReq());
    const body = await res.json();
    expect(body.model_drift).toBeDefined();
    expect(body.model_drift.health).toBe("healthy");
  });

  it("includes alpha decay hint", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => SAMPLE_HEAT,
    });
    const res = await GET(makeReq());
    const body = await res.json();
    expect(body.alpha_decay).toBeDefined();
    expect(typeof body.alpha_decay.optimal_hold_hours).toBe("number");
  });

  it("returns 503 when backend is unreachable", async () => {
    mockFetch.mockRejectedValueOnce(new Error("connection refused"));
    const res = await GET(makeReq());
    expect(res.status).toBe(503);
    const body = await res.json();
    expect(body.error).toMatch(/unavailable/i);
  });

  it("returns backend error status when backend returns 401", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: "Unauthorized" }),
    });
    const res = await GET(makeReq());
    expect(res.status).toBe(401);
  });
});
