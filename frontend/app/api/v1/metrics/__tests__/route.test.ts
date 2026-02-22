/**
 * @jest-environment node
 */
import { NextRequest } from "next/server";
import { jest } from "@jest/globals";

// Mock fetch before importing handler
const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

function makeRequest(token = "test-token"): NextRequest {
  const url = "http://localhost:3000/api/v1/metrics";
  return new NextRequest(url, {
    headers: { cookie: `token=${token}` },
  });
}

function mockUpstream(data: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: async () => data,
    text: async () => JSON.stringify(data),
  } as Response;
}

describe("metrics route — starting_capital passthrough", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("passes through starting_capital from backend", async () => {
    mockFetch.mockResolvedValueOnce(
      mockUpstream({
        status: "online",
        timestamp: "2024-01-01T12:00:00Z",
        capital: 1050000,
        starting_capital: 1000000,
        daily_pnl: 5000,
        total_pnl: 50000,
        max_drawdown: -0.03,
        sharpe_ratio: 1.5,
        win_rate: 0.6,
        open_positions: 5,
        total_trades: 42,
      })
    );

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.starting_capital).toBe(1000000);
    expect(body.capital).toBe(1050000);
  });

  it("derives starting_capital from capital and total_pnl when missing", async () => {
    mockFetch.mockResolvedValueOnce(
      mockUpstream({
        status: "online",
        timestamp: "2024-01-01T12:00:00Z",
        capital: 500000,
        // no starting_capital
        daily_pnl: 1000,
        total_pnl: 10000,
        max_drawdown: -0.01,
        sharpe_ratio: 2.0,
        win_rate: 0.65,
        open_positions: 3,
        total_trades: 20,
      })
    );

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.starting_capital).toBe(490000);
  });

  it("handles non-numeric starting_capital by deriving a stable fallback", async () => {
    mockFetch.mockResolvedValueOnce(
      mockUpstream({
        status: "online",
        timestamp: "2024-01-01T12:00:00Z",
        capital: 500000,
        starting_capital: "not_a_number",
        daily_pnl: 0,
        total_pnl: 0,
        max_drawdown: 0,
        sharpe_ratio: 0,
        win_rate: 0,
        open_positions: 0,
        total_trades: 0,
      })
    );

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.starting_capital).toBe(500000);
  });
});

describe("metrics route — basic behavior", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("returns 401 when no token is present", async () => {
    const req = new NextRequest("http://localhost:3000/api/v1/metrics");
    const response = await GET(req);
    expect(response.status).toBe(401);
  });

  it("maps total_trades to trades_count in response", async () => {
    mockFetch.mockResolvedValueOnce(
      mockUpstream({
        status: "online",
        timestamp: "2024-01-01T12:00:00Z",
        capital: 500000,
        starting_capital: 500000,
        daily_pnl: 0,
        total_pnl: 0,
        max_drawdown: 0,
        sharpe_ratio: 0,
        win_rate: 0,
        open_positions: 0,
        total_trades: 42,
      })
    );

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.trades_count).toBe(42);
  });

  it("returns 503 when upstream fetch throws", async () => {
    mockFetch.mockRejectedValueOnce(new Error("Connection refused"));

    const response = await GET(makeRequest());
    expect(response.status).toBe(503);

    const body = await response.json();
    expect(body.detail).toContain("Connection refused");
  });

  it("forwards upstream error status", async () => {
    mockFetch.mockResolvedValueOnce(
      mockUpstream("Internal Server Error", false, 500)
    );

    const response = await GET(makeRequest());
    expect(response.status).toBe(500);
  });

  it("sanitizes absurd KPI outliers from upstream payload", async () => {
    mockFetch.mockResolvedValueOnce(
      mockUpstream({
        status: "online",
        timestamp: "2024-01-01T12:00:00Z",
        capital: 950000,
        starting_capital: 900000,
        daily_pnl: "oops",
        total_pnl: 50000,
        max_drawdown: -9999,
        sharpe_ratio: "-92962852034076208.00",
        win_rate: 58,
        open_positions: -10,
        total_trades: 8_000_000,
      })
    );

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.capital).toBe(950000);
    expect(body.starting_capital).toBe(900000);
    expect(body.daily_pnl).toBe(0);
    expect(body.max_drawdown).toBe(0);
    expect(body.sharpe_ratio).toBe(0);
    expect(body.win_rate).toBeCloseTo(0.58, 8);
    expect(body.open_positions).toBe(0);
    expect(body.trades_count).toBe(0);
  });
});
