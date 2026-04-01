/**
 * @jest-environment node
 */
import { jest } from "@jest/globals";
import { NextRequest } from "next/server";

const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

const makeReq = (params?: Record<string, string>, token?: string) => {
  const url = new URL("http://localhost/api/v1/order-rejections");
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  return new NextRequest(url, {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });
};

function mockBackend(payload: object | null) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const MOCK_REJECTIONS = {
  available: true,
  total_scanned: 120,
  total_rejected: 3,
  reason_breakdown: { max_order_notional: 2, price_band: 1 },
  rejections: [
    {
      event_id: "ptg-abc123",
      timestamp: "2026-03-23T10:01:00",
      symbol: "AAPL",
      asset_class: "equity",
      side: "BUY",
      quantity: 500,
      price: 180.0,
      reason_code: "max_order_notional",
      message: "Order notional $90,000 exceeds limit $50,000",
      metadata: { notional: 90000, limit: 50000 },
      actor: "strategy_loop",
    },
    {
      event_id: "ptg-def456",
      timestamp: "2026-03-23T09:45:00",
      symbol: "BTC/USD",
      asset_class: "crypto",
      side: "BUY",
      quantity: 1,
      price: 83000.0,
      reason_code: "max_order_notional",
      message: "Order notional $83,000 exceeds limit $50,000",
      metadata: { notional: 83000, limit: 100000 },
      actor: "strategy_loop",
    },
    {
      event_id: "ptg-ghi789",
      timestamp: "2026-03-23T08:30:00",
      symbol: "TSLA",
      asset_class: "equity",
      side: "BUY",
      quantity: 100,
      price: 340.0,
      reason_code: "price_band",
      message: "Price deviation 310.5bps exceeds limit 250.0bps",
      metadata: { deviation_bps: 310.5, limit_bps: 250.0 },
      actor: "strategy_loop",
    },
  ],
};

describe("GET /api/v1/order-rejections", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with available=true and rejections array", async () => {
    mockBackend(MOCK_REJECTIONS);
    const res = await GET(makeReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(Array.isArray(body.rejections)).toBe(true);
  });

  it("passes total_rejected through", async () => {
    mockBackend(MOCK_REJECTIONS);
    const body = await (await GET(makeReq())).json();
    expect(body.total_rejected).toBe(3);
  });

  it("passes reason_breakdown through", async () => {
    mockBackend(MOCK_REJECTIONS);
    const body = await (await GET(makeReq())).json();
    expect(body.reason_breakdown.max_order_notional).toBe(2);
    expect(body.reason_breakdown.price_band).toBe(1);
  });

  it("rejection records have expected fields", async () => {
    mockBackend(MOCK_REJECTIONS);
    const body = await (await GET(makeReq())).json();
    const r = body.rejections[0];
    expect(r).toHaveProperty("event_id");
    expect(r).toHaveProperty("symbol");
    expect(r).toHaveProperty("reason_code");
    expect(r).toHaveProperty("message");
    expect(r).toHaveProperty("timestamp");
  });

  it("passes limit param to backend", async () => {
    mockBackend(MOCK_REJECTIONS);
    await GET(makeReq({ limit: "10" }));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("limit=10"),
      expect.any(Object)
    );
  });

  it("passes reason_code filter to backend", async () => {
    mockBackend(MOCK_REJECTIONS);
    await GET(makeReq({ reason_code: "price_band" }));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("reason_code=price_band"),
      expect.any(Object)
    );
  });

  it("returns available=false when backend unreachable", async () => {
    mockBackend(null);
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(Array.isArray(body.rejections)).toBe(true);
    expect(body.rejections).toHaveLength(0);
  });

  it("handles fetch rejection gracefully", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network timeout"));
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
  });

  it("forwards auth header to backend", async () => {
    mockBackend(MOCK_REJECTIONS);
    await GET(makeReq({}, "tok123"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/order-rejections"),
      expect.objectContaining({
        headers: expect.objectContaining({ authorization: "Bearer tok123" }),
      })
    );
  });
});
