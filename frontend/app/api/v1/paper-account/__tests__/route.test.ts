/**
 * @jest-environment node
 */
import { jest } from "@jest/globals";
import { NextRequest } from "next/server";

const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

const makeReq = (token?: string) =>
  new NextRequest("http://localhost/api/v1/paper-account", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null, ok = true) {
  if (!ok || payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const FULL_SNAPSHOT = {
  available: true,
  open_positions: 2,
  closed_trades: 10,
  paper_total_pnl: 500.0,
  live_total_pnl: 450.0,
  implementation_shortfall_usd: 50.0,
  shortfall_pct: 10.0,
  avg_shortfall_per_trade: 5.0,
  win_rates: { paper: 0.7, live: 0.65, n: 10 },
  day_start_ts: 1700000000,
  recent_trades: [
    {
      symbol: "AAPL",
      side: "BUY",
      entry_price: 175.0,
      exit_price: 177.0,
      notional: 5000,
      pnl_usd: 57.14,
      live_pnl_usd: 50.0,
      shortfall_usd: 7.14,
      entry_ts: 1700000100,
      exit_ts: 1700003600,
    },
  ],
};

afterEach(() => jest.clearAllMocks());

describe("GET /api/v1/paper-account", () => {
  it("returns full snapshot when backend succeeds", async () => {
    mockBackend(FULL_SNAPSHOT);
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(body.implementation_shortfall_usd).toBe(50.0);
    expect(body.win_rates.paper).toBe(0.7);
    expect(body.recent_trades).toHaveLength(1);
  });

  it("forwards auth header to backend", async () => {
    mockBackend(FULL_SNAPSHOT);
    await GET(makeReq("mytoken"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/paper-account"),
      expect.objectContaining({ headers: { authorization: "Bearer mytoken" } })
    );
  });

  it("returns fallback on backend 503", async () => {
    mockBackend(null, false);
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.implementation_shortfall_usd).toBe(0);
    expect(body.recent_trades).toEqual([]);
  });

  it("returns fallback on fetch exception", async () => {
    mockFetch.mockRejectedValueOnce(new Error("connection refused"));
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.note).toMatch(/unreachable/i);
  });

  it("passes through shortfall_pct", async () => {
    mockBackend(FULL_SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.shortfall_pct).toBe(10.0);
  });

  it("passes through avg_shortfall_per_trade", async () => {
    mockBackend(FULL_SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.avg_shortfall_per_trade).toBe(5.0);
  });

  it("passes through paper_total_pnl and live_total_pnl", async () => {
    mockBackend(FULL_SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.paper_total_pnl).toBe(500.0);
    expect(body.live_total_pnl).toBe(450.0);
  });

  it("works without auth token", async () => {
    mockBackend({ available: true, implementation_shortfall_usd: 0, recent_trades: [] });
    const res = await GET(makeReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
  });
});
