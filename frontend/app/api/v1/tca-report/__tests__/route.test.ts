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
  new NextRequest("http://localhost/api/v1/tca-report", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const MOCK_TCA = {
  generated_at: "2026-03-23T10:00:00",
  summary: {
    closed_trades: 42,
    win_rate_pct: 61.9,
    total_net_pnl: 1234.56,
    total_execution_drag: -87.3,
    alpha_before_costs: 1321.86,
    cost_ratio_pct: 6.6,
    total_fills: 84,
    total_rejections: 5,
    rejection_breakdown: { risk_guard: 3, cooldown: 2 },
    execution_health_score: 78.5,
  },
  per_symbol: [
    {
      symbol: "AAPL",
      closed_trades: 10,
      win_rate_pct: 70.0,
      net_pnl: 500.0,
      execution_drag: -20.0,
      avg_entry_slip_bps: 3.2,
      avg_exit_slip_bps: 4.1,
      median_fill_ms: 112,
      p95_fill_ms: 320,
      fills: 10,
      exit_reasons: { take_profit: 6, stop_loss: 2, excellence_exit: 2 },
      open_position: false,
      rejections: {},
    },
  ],
  open_book: {},
};

describe("GET /api/v1/tca-report", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with available=true and summary", async () => {
    mockBackend(MOCK_TCA);
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(body).toHaveProperty("summary");
  });

  it("summary.execution_health_score passed through", async () => {
    mockBackend(MOCK_TCA);
    const body = await (await GET(makeReq())).json();
    expect(body.summary.execution_health_score).toBeCloseTo(78.5);
  });

  it("summary.win_rate_pct passed through", async () => {
    mockBackend(MOCK_TCA);
    const body = await (await GET(makeReq())).json();
    expect(body.summary.win_rate_pct).toBeCloseTo(61.9);
  });

  it("per_symbol array passed through", async () => {
    mockBackend(MOCK_TCA);
    const body = await (await GET(makeReq())).json();
    expect(Array.isArray(body.per_symbol)).toBe(true);
    expect(body.per_symbol[0].symbol).toBe("AAPL");
  });

  it("generated_at passed through", async () => {
    mockBackend(MOCK_TCA);
    const body = await (await GET(makeReq())).json();
    expect(body.generated_at).toBe("2026-03-23T10:00:00");
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
    mockBackend(MOCK_TCA);
    await GET(makeReq("mytoken"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/tca"),
      expect.objectContaining({
        headers: expect.objectContaining({ authorization: "Bearer mytoken" }),
      })
    );
  });
});
