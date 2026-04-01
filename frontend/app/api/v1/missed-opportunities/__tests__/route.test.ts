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
  new NextRequest("http://localhost/api/v1/missed-opportunities", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const MOCK_REPORT = {
  total_missed: 5,
  total_missed_pnl_5d: 0.12,
  total_missed_pnl_10d: 0.18,
  by_filter_reason: { signal_threshold: 3, confidence_threshold: 2 },
  by_regime: { neutral: 0.10, bull: 0.08 },
  top_missed_symbols: [
    { symbol: "NVDA", total_missed_pnl_pct: 0.25 },
    { symbol: "AAPL", total_missed_pnl_pct: 0.08 },
  ],
  generated_at: "2026-01-01T12:00:00Z",
};

const MOCK_PENDING = [
  {
    symbol: "TSLA", signal_strength: 0.17, confidence: 0.60, direction: "long",
    regime: "neutral", filter_reason: "signal_threshold", entry_price: 200.0, asset_class: "equity",
    signal_date: "2026-01-01T10:00:00Z",
  },
];

const MOCK_FULL = {
  pending_count: 3,
  completed_count: 5,
  recent_pending: MOCK_PENDING,
  report: MOCK_REPORT,
};

describe("GET /api/v1/missed-opportunities", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with required keys", async () => {
    mockBackend(MOCK_FULL);
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("pending_count");
    expect(body).toHaveProperty("completed_count");
    expect(body).toHaveProperty("recent_pending");
    expect(body).toHaveProperty("report");
  });

  it("pending_count passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.pending_count).toBe(3);
  });

  it("recent_pending array passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.recent_pending).toHaveLength(1);
    expect(body.recent_pending[0].symbol).toBe("TSLA");
  });

  it("report total_missed passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.report.total_missed).toBe(5);
  });

  it("by_filter_reason passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.report.by_filter_reason.signal_threshold).toBe(3);
  });

  it("top_missed_symbols passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.report.top_missed_symbols[0].symbol).toBe("NVDA");
  });

  it("handles backend failure gracefully", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network error"));
    const body = await (await GET(makeReq())).json();
    expect(body.pending_count).toBe(0);
    expect(body.recent_pending).toEqual([]);
    expect(body.note).toBeTruthy();
  });

  it("missed_pnl_10d passed through", async () => {
    mockBackend(MOCK_FULL);
    const body = await (await GET(makeReq())).json();
    expect(body.report.total_missed_pnl_10d).toBeCloseTo(0.18);
  });
});
