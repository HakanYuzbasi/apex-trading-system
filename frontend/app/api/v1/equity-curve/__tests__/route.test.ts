/**
 * @jest-environment node
 */
// Tests for /api/v1/equity-curve — fetch mocked, no network calls
import { jest } from "@jest/globals";
import { NextRequest } from "next/server";

const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

const makeReq = (token?: string) =>
  new NextRequest("http://localhost/api/v1/equity-curve", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const MOCK_CURVE = [
  { t: "2026-01-01T09:00:00Z", v: 100000 },
  { t: "2026-01-02T09:00:00Z", v: 102000 },
  { t: "2026-01-03T09:00:00Z", v: 101000 },
];
const MOCK_DD = [
  { t: "2026-01-01T09:00:00Z", dd: 0 },
  { t: "2026-01-02T09:00:00Z", dd: 0 },
  { t: "2026-01-03T09:00:00Z", dd: -0.98 },
];

const MOCK_FULL = {
  curve: MOCK_CURVE,
  drawdown: MOCK_DD,
  peak: 102000,
  current: 101000,
  drawdown_pct: -0.98,
  total_points: 3,
};

describe("GET /api/v1/equity-curve", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with required keys", async () => {
    mockBackend(MOCK_FULL);
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("curve");
    expect(body).toHaveProperty("drawdown");
    expect(body).toHaveProperty("peak");
    expect(body).toHaveProperty("current");
    expect(body).toHaveProperty("drawdown_pct");
    expect(body).toHaveProperty("total_points");
  });

  it("curve array passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.curve).toHaveLength(3);
    expect(body.curve[0].v).toBe(100000);
  });

  it("drawdown_pct passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.drawdown_pct).toBeCloseTo(-0.98);
  });

  it("peak passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.peak).toBe(102000);
  });

  it("current passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.current).toBe(101000);
  });

  it("total_points passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.total_points).toBe(3);
  });

  it("handles backend failure — empty curve", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network error"));
    const body = await (await GET(makeReq())).json();
    expect(body.curve).toEqual([]);
    expect(body.drawdown).toEqual([]);
  });

  it("handles null response — note set", async () => {
    mockBackend({ curve: [], drawdown: [], peak: 0, current: 0, drawdown_pct: 0, total_points: 0, note: "engine not running" });
    const body = await (await GET(makeReq())).json();
    expect(body.note).toBe("engine not running");
  });
});
