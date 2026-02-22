/**
 * @jest-environment node
 */
import { NextRequest } from "next/server";
import { jest } from "@jest/globals";

// Mock global fetch
const mockFetch = jest.fn() as jest.MockedFunction<typeof globalThis.fetch>;
globalThis.fetch = mockFetch;

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { GET } = require("../route") as { GET: (req: NextRequest) => Promise<Response> };

function makeRequest(token = "test-token"): NextRequest {
  const url = "http://localhost:3000/api/v1/cockpit";
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

function makeStatusData(overrides: Record<string, unknown> = {}) {
  return {
    status: "online",
    capital: 1000000,
    starting_capital: 1000000,
    daily_pnl: 0,
    total_pnl: 0,
    max_drawdown: -0.02,
    sharpe_ratio: 2.0,
    win_rate: 0.6,
    open_positions: 5,
    total_trades: 50,
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

function mockAllEndpoints(
  statusData: Record<string, unknown>,
  options: {
    daemonPositions?: unknown[];
    statePayload?: Record<string, unknown>;
    portfolioPositions?: unknown[];
    portfolioSources?: unknown[];
    portfolioBalance?: Record<string, unknown>;
  } = {},
) {
  const {
    daemonPositions = [],
    statePayload = { positions: {} },
    portfolioPositions = [],
    portfolioSources = [],
    portfolioBalance = { total_equity: Number(statusData.capital ?? 0), breakdown: [] },
  } = options;
  mockFetch
    .mockResolvedValueOnce(mockUpstream(statusData))         // /status
    .mockResolvedValueOnce(mockUpstream(daemonPositions))    // /positions
    .mockResolvedValueOnce(mockUpstream(statePayload))       // /state
    .mockResolvedValueOnce(mockUpstream({ events: [] }))     // /social-governor
    .mockResolvedValueOnce(mockUpstream(portfolioPositions)) // /portfolio/positions
    .mockResolvedValueOnce(mockUpstream(portfolioSources))   // /portfolio/sources
    .mockResolvedValueOnce(mockUpstream(portfolioBalance));  // /portfolio/balance
}

describe("cockpit route — drawdown alerts", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("fires drawdown alert when normalized drawdown >= 8% (fraction form)", async () => {
    mockAllEndpoints(makeStatusData({ max_drawdown: -0.10 })); // 10% as fraction

    const response = await GET(makeRequest());
    const body = await response.json();

    const alert = body.alerts.find((a: { id: string }) => a.id === "drawdown-warning");
    expect(alert).toBeDefined();
    expect(alert.severity).toBe("warning");
    expect(alert.detail).toContain("-10.00%");
  });

  it("does NOT fire drawdown alert when drawdown < 8%", async () => {
    mockAllEndpoints(makeStatusData({ max_drawdown: -0.05 })); // 5% — under threshold

    const response = await GET(makeRequest());
    const body = await response.json();

    const alert = body.alerts.find((a: { id: string }) => a.id === "drawdown-warning");
    expect(alert).toBeUndefined();
  });

  it("fires drawdown alert when drawdown is in percentage form (>1)", async () => {
    mockAllEndpoints(makeStatusData({ max_drawdown: -9 })); // Already percentage

    const response = await GET(makeRequest());
    const body = await response.json();

    const alert = body.alerts.find((a: { id: string }) => a.id === "drawdown-warning");
    expect(alert).toBeDefined();
  });

  it("fires drawdown alert at the exact boundary of 8%", async () => {
    mockAllEndpoints(makeStatusData({ max_drawdown: -0.08 })); // Exactly 8%

    const response = await GET(makeRequest());
    const body = await response.json();

    const alert = body.alerts.find((a: { id: string }) => a.id === "drawdown-warning");
    expect(alert).toBeDefined();
    expect(alert.detail).toContain("-8.00%");
  });

  it("does NOT fire drawdown alert at 7.99%", async () => {
    mockAllEndpoints(makeStatusData({ max_drawdown: -0.0799 })); // Just under 8%

    const response = await GET(makeRequest());
    const body = await response.json();

    const alert = body.alerts.find((a: { id: string }) => a.id === "drawdown-warning");
    expect(alert).toBeUndefined();
  });
});

describe("cockpit route — starting_capital passthrough", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("includes starting_capital in status response", async () => {
    mockAllEndpoints(makeStatusData({ starting_capital: 1000000 }));

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.status.starting_capital).toBe(1000000);
  });

  it("derives starting_capital when missing from upstream", async () => {
    const data = makeStatusData();
    delete (data as Record<string, unknown>).starting_capital;
    mockAllEndpoints(data);

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.status.starting_capital).toBe(1000000);
  });
});

describe("cockpit route — authentication", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("returns 401 when no token cookie is present", async () => {
    const req = new NextRequest("http://localhost:3000/api/v1/cockpit");
    const response = await GET(req);
    expect(response.status).toBe(401);

    const body = await response.json();
    expect(body.detail).toBe("Not authenticated");
  });
});

describe("cockpit route — sharpe alert", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("fires sharpe alert when below 1.0", async () => {
    mockAllEndpoints(makeStatusData({ sharpe_ratio: 0.8 }));

    const response = await GET(makeRequest());
    const body = await response.json();

    const alert = body.alerts.find((a: { id: string }) => a.id === "sharpe-warning");
    expect(alert).toBeDefined();
    expect(alert.detail).toContain("0.80");
  });

  it("does NOT fire sharpe alert when at or above 1.0", async () => {
    mockAllEndpoints(makeStatusData({ sharpe_ratio: 1.5 }));

    const response = await GET(makeRequest());
    const body = await response.json();

    const alert = body.alerts.find((a: { id: string }) => a.id === "sharpe-warning");
    expect(alert).toBeUndefined();
  });

  it("sanitizes absurd sharpe outliers before alerting and response output", async () => {
    mockAllEndpoints(
      makeStatusData({
        sharpe_ratio: "-92962852034076208.00",
        win_rate: 65,
        open_positions: -3,
      }),
    );

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.status.sharpe_ratio).toBe(0);
    expect(body.status.win_rate).toBeCloseTo(0.65, 8);
    expect(body.status.open_positions).toBe(0);
    const alert = body.alerts.find((a: { id: string }) => a.id === "sharpe-warning");
    expect(alert).toBeDefined();
    expect(alert.detail).toContain("0.00");
  });
});

describe("cockpit route — /portfolio/positions fallback", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("uses /portfolio/positions when state is stale and daemon positions are flat", async () => {
    mockAllEndpoints(
      makeStatusData({ status: "offline", open_positions: 0 }),
      {
        daemonPositions: [],
        statePayload: { positions: {}, open_positions: 0, timestamp: "2020-01-01T00:00:00Z" },
        portfolioPositions: [
          {
            symbol: "AAPL",
            qty: 10,
            side: "LONG",
            avg_cost: 190.12,
            current_price: 194.2,
            unrealized_pl: 40.8,
            unrealized_plpc: 0.021,
            source_id: "ibkr-1",
          },
        ],
      },
    );

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.status.open_positions).toBe(1);
    expect(body.positions).toHaveLength(1);
    expect(body.positions[0]).toMatchObject({
      symbol: "AAPL",
      qty: 10,
      entry: 190.12,
      current: 194.2,
      pnl: 40.8,
      source_id: "ibkr-1",
    });
    const fallbackAlert = body.alerts.find((a: { id: string }) => a.id === "portfolio-position-fallback");
    expect(fallbackAlert).toBeDefined();
  });

  it("keeps portfolio aggregation as primary when portfolio rows are present", async () => {
    mockAllEndpoints(
      makeStatusData({ status: "online", open_positions: 2 }),
      {
        daemonPositions: [
          {
            symbol: "MSFT",
            qty: 2,
            side: "LONG",
            entry: 410,
            current: 411,
            pnl: 2,
            pnl_pct: 0.0048,
          },
        ],
        statePayload: { positions: { MSFT: { qty: 2 } }, open_positions: 2, timestamp: new Date().toISOString() },
        portfolioPositions: [
          {
            symbol: "AAPL",
            qty: 10,
            side: "LONG",
            avg_cost: 190,
            current_price: 194,
            unrealized_pl: 40,
            unrealized_plpc: 0.02,
          },
        ],
      },
    );

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.positions).toHaveLength(1);
    expect(body.positions[0].symbol).toBe("AAPL");
    const fallbackAlert = body.alerts.find((a: { id: string }) => a.id === "portfolio-position-fallback");
    expect(fallbackAlert).toBeDefined();
  });
});
