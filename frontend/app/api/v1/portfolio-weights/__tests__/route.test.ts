/**
 * @jest-environment node
 */
import { jest } from "@jest/globals";
import { NextRequest } from "next/server";

const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

function makeReq(token?: string): NextRequest {
  return new NextRequest("http://localhost/api/v1/portfolio-weights", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });
}

function mockBackendFetch(data: unknown, ok = true) {
  if (!ok || data === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => data } as Response);
  }
}

afterEach(() => jest.clearAllMocks());

describe("GET /api/v1/portfolio-weights", () => {
  it("returns available:true with weights when backend succeeds", async () => {
    const payload = {
      available: true,
      method: "hrp_signal",
      n_symbols: 5,
      weights: { AAPL: 0.15, MSFT: 0.12 },
      top_signals: { AAPL: 0.22, MSFT: 0.18 },
    };
    mockBackendFetch(payload);
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(body.weights).toEqual({ AAPL: 0.15, MSFT: 0.12 });
  });

  it("forwards auth header to backend", async () => {
    mockBackendFetch({ available: true, weights: {} });
    await GET(makeReq("mytoken"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/portfolio-weights"),
      expect.objectContaining({ headers: { authorization: "Bearer mytoken" } })
    );
  });

  it("returns available:false when backend returns non-ok", async () => {
    mockBackendFetch(null, false);
    const res = await GET(makeReq());
    const body = await res.json();
    expect(body.available).toBe(false);
    expect(body.weights).toEqual({});
  });

  it("returns available:false when fetch throws", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network error"));
    const res = await GET(makeReq());
    const body = await res.json();
    expect(body.available).toBe(false);
    expect(body.note).toMatch(/unreachable/i);
  });

  it("passes through method field", async () => {
    mockBackendFetch({ available: true, method: "equal", weights: {} });
    const res = await GET(makeReq());
    const body = await res.json();
    expect(body.method).toBe("equal");
  });

  it("passes through cov_condition field", async () => {
    mockBackendFetch({ available: true, cov_condition: 42.5, weights: {} });
    const res = await GET(makeReq());
    const body = await res.json();
    expect(body.cov_condition).toBe(42.5);
  });

  it("works without auth token (empty headers)", async () => {
    mockBackendFetch({ available: true, weights: { BTC: 0.10 } });
    const res = await GET(makeReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.weights.BTC).toBe(0.10);
  });

  it("passes through top_signals", async () => {
    mockBackendFetch({ available: true, weights: {}, top_signals: { TSLA: 0.21 } });
    const res = await GET(makeReq());
    const body = await res.json();
    expect(body.top_signals.TSLA).toBe(0.21);
  });
});
