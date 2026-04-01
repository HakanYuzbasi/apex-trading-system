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
  new NextRequest("http://localhost/api/v1/stress-state", {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });

function mockBackend(payload: object | null) {
  if (payload === null) {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, json: async () => null } as Response);
  } else {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => payload } as Response);
  }
}

const MOCK_STATE = {
  enabled: true,
  active: true,
  evaluated_at: "2026-01-01T12:00:00Z",
  scenario_count: 3,
  action: "warn",
  halt_new_entries: false,
  size_multiplier: 0.6,
  reason: "Warning threshold breached",
  worst_scenario_id: "crash_2020",
  worst_scenario_name: "COVID Crash",
  worst_portfolio_return: -0.12,
  worst_portfolio_pnl: -12000,
  worst_drawdown: 0.15,
  breached_limits: ["drawdown_limit"],
  recommendations: ["Reduce position sizes"],
  scenarios: [],
};

describe("GET /api/v1/stress-state", () => {
  beforeEach(() => mockFetch.mockReset());

  it("returns 200 with state key", async () => {
    mockBackend({ state: MOCK_STATE });
    const res = await GET(makeReq("tok"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty("state");
  });

  it("state action passed through", async () => {
    mockBackend({ state: MOCK_STATE });
    const body = await (await GET(makeReq())).json();
    expect(body.state.action).toBe("warn");
  });

  it("halt_new_entries passed through", async () => {
    mockBackend({ state: { ...MOCK_STATE, halt_new_entries: true } });
    const body = await (await GET(makeReq())).json();
    expect(body.state.halt_new_entries).toBe(true);
  });

  it("size_multiplier passed through", async () => {
    mockBackend({ state: MOCK_STATE });
    const body = await (await GET(makeReq())).json();
    expect(body.state.size_multiplier).toBeCloseTo(0.6);
  });

  it("worst_scenario_name passed through", async () => {
    mockBackend({ state: MOCK_STATE });
    const body = await (await GET(makeReq())).json();
    expect(body.state.worst_scenario_name).toBe("COVID Crash");
  });

  it("returns null state when backend returns null state", async () => {
    mockBackend({ state: null, note: "engine not running" });
    const body = await (await GET(makeReq())).json();
    expect(body.state).toBeNull();
    expect(body.note).toBe("engine not running");
  });

  it("handles network failure gracefully", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network error"));
    const body = await (await GET(makeReq())).json();
    expect(body.state).toBeNull();
  });

  it("breached_limits passed through", async () => {
    mockBackend({ state: MOCK_STATE });
    const body = await (await GET(makeReq())).json();
    expect(body.state.breached_limits).toContain("drawdown_limit");
  });
});
