/**
 * @jest-environment node
 */
// Tests for /api/v1/trade-postmortem — fetch mocked, no network calls
import { jest } from "@jest/globals";
import { NextRequest } from "next/server";

const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

const makeReq = (token?: string) =>
  new NextRequest("http://localhost/api/v1/trade-postmortem", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({
      ok: false, status: 503, json: async () => null,
    } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({
      ok: true, status: 200, json: async () => payload,
    } as Response);
  }
}

const MOCK_RECENT = [
  {
    symbol: "AAPL", pnl_pct: 0.025, hold_hours: 4.0, exit_reason: "signal",
    signal_quality: "good", timing: "good", regime_alignment: "good",
    execution_drag: "good", verdict: "winner", primary_failure: "none",
    confidence_at_entry: 0.72, signal_at_entry: 0.22,
    slippage_bps: 3.0, regime: "bull", timestamp: "2026-01-01T12:00:00Z",
  },
];

const MOCK_SUMMARY = {
  total: 10,
  win_rate: 0.7,
  avg_pnl_pct: 0.015,
  verdict_counts: { winner: 7, loser: 2, breakeven: 1 },
  failure_counts: { weak_signal: 2, none: 7, unknown: 1 },
};

describe("GET /api/v1/trade-postmortem", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with recent and summary", async () => {
    mockBackend({ recent: MOCK_RECENT, summary: MOCK_SUMMARY });
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("recent");
    expect(body).toHaveProperty("summary");
  });

  it("recent array is passed through", async () => {
    mockBackend({ recent: MOCK_RECENT, summary: MOCK_SUMMARY });
    const body = await (await GET(makeReq())).json();
    expect(body.recent).toHaveLength(1);
    expect(body.recent[0].symbol).toBe("AAPL");
    expect(body.recent[0].verdict).toBe("winner");
  });

  it("summary win_rate passed through", async () => {
    mockBackend({ recent: [], summary: MOCK_SUMMARY });
    const body = await (await GET(makeReq())).json();
    expect(body.summary.win_rate).toBeCloseTo(0.7);
    expect(body.summary.total).toBe(10);
  });

  it("verdict_counts passed through", async () => {
    mockBackend({ recent: [], summary: MOCK_SUMMARY });
    const body = await (await GET(makeReq())).json();
    expect(body.summary.verdict_counts.winner).toBe(7);
    expect(body.summary.verdict_counts.loser).toBe(2);
  });

  it("handles backend failure gracefully — empty recent", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network error"));
    const body = await (await GET(makeReq())).json();
    expect(body.recent).toEqual([]);
    expect(body.note).toBeTruthy();
  });

  it("note field is passed through when present", async () => {
    mockBackend({ recent: [], summary: {}, note: "engine not running" });
    const body = await (await GET(makeReq())).json();
    expect(body.note).toBe("engine not running");
  });

  it("empty summary returns zero total", async () => {
    mockBackend({ recent: [], summary: { total: 0, win_rate: 0, avg_pnl_pct: 0, verdict_counts: {}, failure_counts: {} } });
    const body = await (await GET(makeReq())).json();
    expect(body.summary.total).toBe(0);
  });

  it("failure_counts passed through", async () => {
    mockBackend({ recent: [], summary: MOCK_SUMMARY });
    const body = await (await GET(makeReq())).json();
    expect(body.summary.failure_counts.weak_signal).toBe(2);
  });
});
