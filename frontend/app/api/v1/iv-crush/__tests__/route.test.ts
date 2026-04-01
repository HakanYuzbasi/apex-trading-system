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
  new NextRequest("http://localhost/api/v1/iv-crush", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null, ok = true) {
  if (!ok || payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const SNAPSHOT = {
  available: true,
  total_tracked: 5,
  active_signals: [
    {
      symbol: "AAPL",
      days_to_earnings: 3,
      iv_elevation: 1.6,
      signal: -0.10,
      confidence: 0.72,
      strategy: "iv_crush",
      earnings_date: "2026-04-25",
    },
  ],
  upcoming_earnings: [
    { symbol: "NVDA", days_to_earnings: 2, iv_elevation: 1.45, signal: -0.08, strategy: "iv_crush" },
  ],
  iv_elevation_threshold: 1.4,
  iv_crush_scale: 0.12,
  pead_scale: 0.15,
};

afterEach(() => jest.clearAllMocks());

describe("GET /api/v1/iv-crush", () => {
  it("returns snapshot when backend succeeds", async () => {
    mockBackend(SNAPSHOT);
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(body.total_tracked).toBe(5);
    expect(body.active_signals).toHaveLength(1);
  });

  it("forwards auth header to backend", async () => {
    mockBackend(SNAPSHOT);
    await GET(makeReq("mytoken"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/iv-crush"),
      expect.objectContaining({ headers: { authorization: "Bearer mytoken" } })
    );
  });

  it("returns fallback on backend 503", async () => {
    mockBackend(null, false);
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.active_signals).toEqual([]);
    expect(body.upcoming_earnings).toEqual([]);
  });

  it("returns fallback on fetch exception", async () => {
    mockFetch.mockRejectedValueOnce(new Error("timeout"));
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.note).toMatch(/unreachable/i);
  });

  it("passes through iv_elevation_threshold", async () => {
    mockBackend(SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.iv_elevation_threshold).toBe(1.4);
  });

  it("passes through upcoming_earnings", async () => {
    mockBackend(SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.upcoming_earnings).toHaveLength(1);
    expect(body.upcoming_earnings[0].symbol).toBe("NVDA");
  });

  it("passes through signal strategy type", async () => {
    mockBackend(SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.active_signals[0].strategy).toBe("iv_crush");
  });

  it("works without auth token", async () => {
    mockBackend({ available: true, total_tracked: 0, active_signals: [], upcoming_earnings: [] });
    const res = await GET(makeReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
  });
});
