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
  new NextRequest("http://localhost/api/v1/cross-asset-pairs", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null, ok = true) {
  if (!ok || payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const PAIR_SNAPSHOT = {
  available: true,
  n_pairs: 2,
  last_scan_ts: 1700000000,
  active_pairs: [
    { leg_y: "BTC/USD", leg_x: "ETH/USD", hedge_ratio: 15.0, half_life: 6.5, z_score: 2.1, signal_y: -0.12, signal_x: 0.10 },
    { leg_y: "MSTR", leg_x: "BTC/USD", hedge_ratio: 0.0002, half_life: 4.0, z_score: -1.9, signal_y: 0.08, signal_x: -0.06 },
  ],
  z_entry: 1.8,
  z_exit: 0.4,
};

afterEach(() => jest.clearAllMocks());

describe("GET /api/v1/cross-asset-pairs", () => {
  it("returns snapshot when backend succeeds", async () => {
    mockBackend(PAIR_SNAPSHOT);
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(body.n_pairs).toBe(2);
    expect(body.active_pairs).toHaveLength(2);
  });

  it("forwards auth header to backend", async () => {
    mockBackend(PAIR_SNAPSHOT);
    await GET(makeReq("mytoken"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/cross-asset-pairs"),
      expect.objectContaining({ headers: { authorization: "Bearer mytoken" } })
    );
  });

  it("returns fallback on backend 503", async () => {
    mockBackend(null, false);
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.n_pairs).toBe(0);
    expect(body.active_pairs).toEqual([]);
  });

  it("returns fallback on fetch exception", async () => {
    mockFetch.mockRejectedValueOnce(new Error("timeout"));
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
    expect(body.note).toMatch(/unreachable/i);
  });

  it("passes through z_entry and z_exit", async () => {
    mockBackend(PAIR_SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.z_entry).toBe(1.8);
    expect(body.z_exit).toBe(0.4);
  });

  it("passes through pair z_score and signals", async () => {
    mockBackend(PAIR_SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.active_pairs[0].z_score).toBe(2.1);
    expect(body.active_pairs[0].signal_y).toBe(-0.12);
  });

  it("passes through last_scan_ts", async () => {
    mockBackend(PAIR_SNAPSHOT);
    const body = await (await GET(makeReq())).json();
    expect(body.last_scan_ts).toBe(1700000000);
  });

  it("works without auth token", async () => {
    mockBackend({ available: true, n_pairs: 0, active_pairs: [] });
    const res = await GET(makeReq());
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
  });
});
