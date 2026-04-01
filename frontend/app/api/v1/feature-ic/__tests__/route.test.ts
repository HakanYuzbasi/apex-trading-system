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
  new NextRequest("http://localhost/api/v1/feature-ic", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const MOCK_IC = {
  available: true,
  n_features: 3,
  pending_snapshots: 12,
  dead_count: 1,
  strong_count: 1,
  dead: ["volume_ratio"],
  strong: ["rsi_14"],
  features: [
    { feature: "rsi_14", ic_30d: 0.072, ic_90d: 0.065, n_obs: 120, status: "strong" },
    { feature: "macd_hist", ic_30d: 0.031, ic_90d: 0.028, n_obs: 95, status: "live" },
    { feature: "volume_ratio", ic_30d: 0.008, ic_90d: 0.005, n_obs: 80, status: "dead" },
  ],
  thresholds: { dead: 0.015, suspect: 0.03, strong: 0.05 },
};

describe("GET /api/v1/feature-ic", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with available=true and features array", async () => {
    mockBackend(MOCK_IC);
    const res = await GET(makeReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(Array.isArray(body.features)).toBe(true);
  });

  it("passes n_features through", async () => {
    mockBackend(MOCK_IC);
    const body = await (await GET(makeReq())).json();
    expect(body.n_features).toBe(3);
  });

  it("passes dead and strong arrays through", async () => {
    mockBackend(MOCK_IC);
    const body = await (await GET(makeReq())).json();
    expect(body.dead).toContain("volume_ratio");
    expect(body.strong).toContain("rsi_14");
  });

  it("feature records have required fields", async () => {
    mockBackend(MOCK_IC);
    const body = await (await GET(makeReq())).json();
    const f = body.features[0];
    expect(f).toHaveProperty("feature");
    expect(f).toHaveProperty("ic_30d");
    expect(f).toHaveProperty("ic_90d");
    expect(f).toHaveProperty("status");
  });

  it("passes thresholds through", async () => {
    mockBackend(MOCK_IC);
    const body = await (await GET(makeReq())).json();
    expect(body.thresholds.strong).toBeCloseTo(0.05);
  });

  it("returns available=false when backend unreachable", async () => {
    mockBackend(null);
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
  });

  it("handles fetch rejection gracefully", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network error"));
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
  });

  it("forwards auth header to backend", async () => {
    mockBackend(MOCK_IC);
    await GET(makeReq("tok99"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/feature-ic"),
      expect.objectContaining({
        headers: expect.objectContaining({ authorization: "Bearer tok99" }),
      })
    );
  });
});
