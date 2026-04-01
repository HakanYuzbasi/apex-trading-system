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
  new NextRequest("http://localhost/api/v1/attribution-summary", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const MOCK_SUMMARY = {
  lookback_days: 30,
  closed_trades: 15,
  gross_pnl: 3500.0,
  net_pnl: 3000.0,
  commissions: 250.0,
  modeled_execution_drag: 100.0,
  modeled_slippage_drag: 150.0,
  by_sleeve: { ibkr: { trades: 10, net_pnl: 2000.0, avg_net_pnl: 200.0 } },
  by_asset_class: { EQUITY: { trades: 12, net_pnl: 2500.0, avg_net_pnl: 208.3 } },
};

const MOCK_SIGNAL_SOURCES = {
  lookback_days: 30,
  by_signal_source: {
    ml: { trades: 8, wins: 6, total_pnl: 1500.0, win_rate: 0.75, avg_net_pnl: 187.5, avg_pnl_bps: 45.0, avg_holding_hours: 3.2 },
    technical: { trades: 7, wins: 4, total_pnl: 1500.0, win_rate: 0.57, avg_net_pnl: 214.3, avg_pnl_bps: 52.0, avg_holding_hours: 4.1 },
  },
};

describe("GET /api/v1/attribution-summary", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with summary and signal_sources", async () => {
    mockBackend({ summary: MOCK_SUMMARY, signal_sources: MOCK_SIGNAL_SOURCES });
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("summary");
    expect(body).toHaveProperty("signal_sources");
  });

  it("summary closed_trades passed through", async () => {
    mockBackend({ summary: MOCK_SUMMARY, signal_sources: MOCK_SIGNAL_SOURCES });
    const body = await (await GET(makeReq())).json();
    expect(body.summary.closed_trades).toBe(15);
  });

  it("net_pnl passed through", async () => {
    mockBackend({ summary: MOCK_SUMMARY, signal_sources: MOCK_SIGNAL_SOURCES });
    const body = await (await GET(makeReq())).json();
    expect(body.summary.net_pnl).toBeCloseTo(3000.0);
  });

  it("signal source ml present", async () => {
    mockBackend({ summary: MOCK_SUMMARY, signal_sources: MOCK_SIGNAL_SOURCES });
    const body = await (await GET(makeReq())).json();
    expect(body.signal_sources.by_signal_source).toHaveProperty("ml");
  });

  it("ml win_rate passed through", async () => {
    mockBackend({ summary: MOCK_SUMMARY, signal_sources: MOCK_SIGNAL_SOURCES });
    const body = await (await GET(makeReq())).json();
    expect(body.signal_sources.by_signal_source.ml.win_rate).toBeCloseTo(0.75);
  });

  it("handles backend failure gracefully", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network error"));
    const body = await (await GET(makeReq())).json();
    expect(body.summary).toEqual({});
    expect(body.note).toBeTruthy();
  });

  it("by_asset_class passed through", async () => {
    mockBackend({ summary: MOCK_SUMMARY, signal_sources: MOCK_SIGNAL_SOURCES });
    const body = await (await GET(makeReq())).json();
    expect(body.summary.by_asset_class).toHaveProperty("EQUITY");
  });

  it("technical source trades count correct", async () => {
    mockBackend({ summary: MOCK_SUMMARY, signal_sources: MOCK_SIGNAL_SOURCES });
    const body = await (await GET(makeReq())).json();
    expect(body.signal_sources.by_signal_source.technical.trades).toBe(7);
  });
});
