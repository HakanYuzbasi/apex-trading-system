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
  new NextRequest("http://localhost/api/v1/stress-scenarios", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const MOCK_SCENARIOS = {
  scenarios: [
    {
      scenario_id: "2008_financial_crisis",
      scenario_name: "2008 Financial Crisis",
      scenario_type: "market_crash",
      portfolio_pnl: -48000,
      portfolio_return_pct: -24.0,
      max_drawdown_pct: -30.0,
      var_95_stressed: -12000,
      expected_shortfall: -18000,
      worst_positions: [{ symbol: "AAPL", pnl: -8000 }],
      breached_limits: ["daily_loss_limit", "max_drawdown"],
      estimated_liquidation_cost: 2400,
      recommendations: ["Reduce equity exposure", "Add hedges"],
    },
  ],
  capital: 200000,
  n_positions: 5,
};

describe("GET /api/v1/stress-scenarios", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with available=true and scenarios array", async () => {
    mockBackend(MOCK_SCENARIOS);
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.available).toBe(true);
    expect(Array.isArray(body.scenarios)).toBe(true);
  });

  it("scenario portfolio_pnl passed through", async () => {
    mockBackend(MOCK_SCENARIOS);
    const body = await (await GET(makeReq())).json();
    expect(body.scenarios[0].portfolio_pnl).toBe(-48000);
  });

  it("capital passed through", async () => {
    mockBackend(MOCK_SCENARIOS);
    const body = await (await GET(makeReq())).json();
    expect(body.capital).toBe(200000);
  });

  it("n_positions passed through", async () => {
    mockBackend(MOCK_SCENARIOS);
    const body = await (await GET(makeReq())).json();
    expect(body.n_positions).toBe(5);
  });

  it("returns available=false when backend unreachable", async () => {
    mockBackend(null);
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
  });

  it("handles fetch rejection gracefully", async () => {
    mockFetch.mockRejectedValueOnce(new Error("conn refused"));
    const body = await (await GET(makeReq())).json();
    expect(body.available).toBe(false);
  });

  it("forwards auth header", async () => {
    mockBackend(MOCK_SCENARIOS);
    await GET(makeReq("secrettoken"));
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/ops/stress-scenarios"),
      expect.objectContaining({
        headers: expect.objectContaining({ authorization: "Bearer secrettoken" }),
      })
    );
  });

  it("scenario_name passed through", async () => {
    mockBackend(MOCK_SCENARIOS);
    const body = await (await GET(makeReq())).json();
    expect(body.scenarios[0].scenario_name).toBe("2008 Financial Crisis");
  });
});
