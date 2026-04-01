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
  new NextRequest("http://localhost/api/v1/advanced-metrics", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const MOCK_METRICS = {
  available: true,
  n_returns: 120,
  cvar_95: -0.018,
  cvar_99: -0.031,
  var_95: -0.012,
  var_99: -0.025,
  sortino_ratio: 1.42,
  calmar_ratio: 0.87,
  omega_ratio: 1.35,
  downside_deviation: 0.008,
  tail_ratio: 1.21,
  skewness: -0.34,
  kurtosis: 3.12,
  max_dd_duration: 18,
};

describe("GET /api/v1/advanced-metrics", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with required keys", async () => {
    mockBackend(MOCK_METRICS);
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("available");
  });

  it("sortino_ratio passed through", async () => {
    mockBackend(MOCK_METRICS);
    const body = await (await GET(makeReq())).json();
    expect(body.sortino_ratio).toBeCloseTo(1.42);
  });

  it("cvar_95 passed through", async () => {
    mockBackend(MOCK_METRICS);
    const body = await (await GET(makeReq())).json();
    expect(body.cvar_95).toBeCloseTo(-0.018);
  });

  it("calmar_ratio passed through", async () => {
    mockBackend(MOCK_METRICS);
    const body = await (await GET(makeReq())).json();
    expect(body.calmar_ratio).toBeCloseTo(0.87);
  });

  it("n_returns passed through", async () => {
    mockBackend(MOCK_METRICS);
    const body = await (await GET(makeReq())).json();
    expect(body.n_returns).toBe(120);
  });

  it("returns available=false when engine not running", async () => {
    mockBackend({ available: false, note: "engine not running" });
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
  });

  it("handles backend failure gracefully", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network error"));
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
  });

  it("omega_ratio passed through", async () => {
    mockBackend(MOCK_METRICS);
    const body = await (await GET(makeReq())).json();
    expect(body.omega_ratio).toBeCloseTo(1.35);
  });
});
