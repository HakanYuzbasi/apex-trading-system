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

describe("cockpit route — broker/source inference stability", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("infers broker readiness from portfolio positions when sources endpoint is empty", async () => {
    mockAllEndpoints(
      makeStatusData({
        primary_execution_broker: "alpaca",
        broker_mode: "both",
        open_positions: 1,
      }),
      {
        portfolioSources: [],
        portfolioBalance: { total_equity: 250000, breakdown: [] },
        portfolioPositions: [
          {
            symbol: "TSLA",
            qty: 4,
            side: "LONG",
            avg_cost: 210,
            current_price: 215,
            unrealized_pl: 20,
            unrealized_plpc: 0.024,
            broker_type: "ibkr",
            source_id: "ibkr-source-1",
          },
        ],
      },
    );

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(body.status.active_broker).toBe("alpaca");
    const ibkr = body.status.brokers.find((row: { broker: string }) => row.broker === "ibkr");
    const alpaca = body.status.brokers.find((row: { broker: string }) => row.broker === "alpaca");
    expect(ibkr).toBeDefined();
    expect(ibkr.configured).toBe(true);
    expect(alpaca).toBeDefined();
    expect(alpaca.configured).toBe(true);
  });
});

describe("cockpit route — social audit cache stability", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("keeps last good social audit snapshot when upstream temporarily fails", async () => {
    const status = makeStatusData({ primary_execution_broker: "alpaca" });
    mockFetch
      .mockResolvedValueOnce(mockUpstream(status)) // /status
      .mockResolvedValueOnce(mockUpstream([])) // /positions
      .mockResolvedValueOnce(mockUpstream({ positions: {} })) // /state
      .mockResolvedValueOnce(mockUpstream({
        count: 1,
        events: [
          {
            audit_id: "social-1",
            timestamp: "2026-02-22T12:00:00Z",
            asset_class: "EQUITY",
            regime: "neutral",
            policy_version: "v1",
            decision_hash: "hash-1",
            decision: {
              block_new_entries: false,
              gross_exposure_multiplier: 1.0,
              combined_risk_score: 0.15,
              verified_event_probability: 0.33,
              prediction_verification_failures: 0,
              reasons: ["ok"],
            },
            verified_events: [],
          },
        ],
      })) // /social-governor
      .mockResolvedValueOnce(mockUpstream([])) // /portfolio/positions
      .mockResolvedValueOnce(mockUpstream([])) // /portfolio/sources
      .mockResolvedValueOnce(mockUpstream({ total_equity: 100000, breakdown: [] })); // /portfolio/balance

    const first = await GET(makeRequest());
    const firstBody = await first.json();
    expect(firstBody.social_audit.available).toBe(true);
    expect(firstBody.social_audit.events).toHaveLength(1);

    mockFetch
      .mockResolvedValueOnce(mockUpstream(status)) // /status
      .mockResolvedValueOnce(mockUpstream([])) // /positions
      .mockResolvedValueOnce(mockUpstream({ positions: {} })) // /state
      .mockResolvedValueOnce(mockUpstream({ detail: "upstream error" }, false, 500)) // /social-governor
      .mockResolvedValueOnce(mockUpstream([])) // /portfolio/positions
      .mockResolvedValueOnce(mockUpstream([])) // /portfolio/sources
      .mockResolvedValueOnce(mockUpstream({ total_equity: 100000, breakdown: [] })); // /portfolio/balance

    const second = await GET(makeRequest());
    const secondBody = await second.json();
    expect(secondBody.social_audit.available).toBe(true);
    expect(secondBody.social_audit.cached).toBe(true);
    expect(secondBody.social_audit.events).toHaveLength(1);
    expect(secondBody.social_audit.warning).toContain("cached");
  });

  it("falls back to state social_shock decisions when social audit endpoint is unavailable", async () => {
    const status = makeStatusData({ primary_execution_broker: "alpaca" });
    mockFetch
      .mockResolvedValueOnce(mockUpstream(status)) // /status
      .mockResolvedValueOnce(mockUpstream([])) // /positions
      .mockResolvedValueOnce(mockUpstream({
        positions: {},
        social_shock: {
          decisions: [
            {
              audit_id: "state-fallback-1",
              timestamp: "2026-02-22T12:00:00Z",
              asset_class: "CRYPTO",
              regime: "trend",
              policy_version: "runtime-v1",
              decision_hash: "state-hash-1",
              block_new_entries: false,
              gross_exposure_multiplier: 0.85,
              combined_risk_score: 0.66,
              verified_event_probability: 0.62,
              prediction_verification_failures: 0,
              verified_event_count: 2,
              reasons: ["social attention spike"],
            },
          ],
        },
      })) // /state
      .mockResolvedValueOnce(mockUpstream({ detail: "downstream unavailable" }, false, 500)) // /social-governor
      .mockResolvedValueOnce(mockUpstream([])) // /portfolio/positions
      .mockResolvedValueOnce(mockUpstream([])) // /portfolio/sources
      .mockResolvedValueOnce(mockUpstream({ total_equity: 100000, breakdown: [] })); // /portfolio/balance

    const response = await GET(makeRequest("state-social-fallback"));
    const body = await response.json();
    expect(body.social_audit.available).toBe(true);
    expect(body.social_audit.events).toHaveLength(1);
    expect(body.social_audit.events[0].audit_id).toBe("state-fallback-1");
    expect(String(body.social_audit.warning)).toContain("state_fallback");
  });
});

describe("cockpit route — position snapshot stability", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("keeps last good positions when /portfolio/positions is transiently unavailable", async () => {
    const token = "position-cache-token";
    const status = makeStatusData({
      status: "online",
      open_positions: 1,
      timestamp: new Date().toISOString(),
    });

    mockFetch
      .mockResolvedValueOnce(mockUpstream(status)) // /status
      .mockResolvedValueOnce(mockUpstream([])) // /positions
      .mockResolvedValueOnce(mockUpstream({ positions: {}, open_positions: 1, timestamp: new Date().toISOString() })) // /state
      .mockResolvedValueOnce(mockUpstream({ events: [] })) // /social-governor
      .mockResolvedValueOnce(mockUpstream([
        {
          symbol: "AAPL",
          qty: 10,
          side: "LONG",
          avg_cost: 190,
          current_price: 194,
          unrealized_pl: 40,
          unrealized_plpc: 0.02,
          source_id: "ibkr-1",
          broker_type: "ibkr",
        },
      ])) // /portfolio/positions
      .mockResolvedValueOnce(mockUpstream([])) // /portfolio/sources
      .mockResolvedValueOnce(mockUpstream({ total_equity: 100000, breakdown: [] })); // /portfolio/balance

    const first = await GET(makeRequest(token));
    const firstBody = await first.json();
    expect(firstBody.positions).toHaveLength(1);
    expect(firstBody.positions[0].symbol).toBe("AAPL");

    mockFetch
      .mockResolvedValueOnce(mockUpstream(status)) // /status
      .mockResolvedValueOnce(mockUpstream([])) // /positions
      .mockResolvedValueOnce(mockUpstream({ positions: {}, open_positions: 1, timestamp: new Date().toISOString() })) // /state
      .mockResolvedValueOnce(mockUpstream({ events: [] })) // /social-governor
      .mockResolvedValueOnce(mockUpstream({ detail: "temporary outage" }, false, 500)) // /portfolio/positions
      .mockResolvedValueOnce(mockUpstream([])) // /portfolio/sources
      .mockResolvedValueOnce(mockUpstream({ total_equity: 100000, breakdown: [] })); // /portfolio/balance

    const second = await GET(makeRequest(token));
    const secondBody = await second.json();
    expect(secondBody.positions).toHaveLength(1);
    expect(secondBody.positions[0].symbol).toBe("AAPL");
    const cacheAlert = secondBody.alerts.find((a: { id: string }) => a.id === "position-cache-fallback");
    expect(cacheAlert).toBeDefined();
  });
});

describe("cockpit route — daily pnl broker-fill source", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("does not overwrite broker-fill daily pnl with inferred unrealized fallback", async () => {
    mockAllEndpoints(
      makeStatusData({
        daily_pnl: 0,
        daily_pnl_realized: 0,
        daily_pnl_source: "broker_fills",
        open_positions: 1,
      }),
      {
        daemonPositions: [],
        statePayload: { positions: {}, open_positions: 1, timestamp: new Date().toISOString() },
        portfolioPositions: [
          {
            symbol: "BTC/USD",
            qty: 1,
            side: "LONG",
            avg_cost: 90000,
            current_price: 90500,
            unrealized_pl: 500,
            unrealized_plpc: 0.0055,
            source_id: "alpaca-1",
            broker_type: "alpaca",
          },
        ],
      },
    );

    const response = await GET(makeRequest("daily-pnl-fill-source"));
    const body = await response.json();
    expect(body.status.daily_pnl).toBe(0);
    expect(body.status.daily_pnl_realized).toBe(0);
    expect(body.status.daily_pnl_source).toBe("broker_fills");
  });
});

describe("cockpit route — position parity watchdog", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("alerts only after mismatch persists for configured streak", async () => {
    const token = "position-parity-streak";
    const status = makeStatusData({
      open_positions: 5,
      timestamp: new Date().toISOString(),
    });
    const mockMismatchCycle = () => {
      mockFetch
        .mockResolvedValueOnce(mockUpstream(status)) // /status
        .mockResolvedValueOnce(mockUpstream([])) // /positions
        .mockResolvedValueOnce(mockUpstream({ positions: {}, open_positions: 5, timestamp: new Date().toISOString() })) // /state
        .mockResolvedValueOnce(mockUpstream({ events: [] })) // /social-governor
        .mockResolvedValueOnce(mockUpstream([
          {
            symbol: "AAPL",
            qty: 1,
            side: "LONG",
            avg_cost: 190,
            current_price: 191,
            unrealized_pl: 1,
            unrealized_plpc: 0.005,
            source_id: "ibkr-1",
            broker_type: "ibkr",
          },
        ])) // /portfolio/positions
        .mockResolvedValueOnce(mockUpstream([])) // /portfolio/sources
        .mockResolvedValueOnce(mockUpstream({ total_equity: 100000, breakdown: [] })); // /portfolio/balance
    };

    mockMismatchCycle();
    const first = await GET(makeRequest(token));
    const firstBody = await first.json();
    expect(firstBody.alerts.find((a: { id: string }) => a.id === "position-parity-watchdog")).toBeUndefined();

    mockMismatchCycle();
    const second = await GET(makeRequest(token));
    const secondBody = await second.json();
    expect(secondBody.alerts.find((a: { id: string }) => a.id === "position-parity-watchdog")).toBeUndefined();

    mockMismatchCycle();
    const third = await GET(makeRequest(token));
    const thirdBody = await third.json();
    const parityAlert = thirdBody.alerts.find((a: { id: string }) => a.id === "position-parity-watchdog");
    expect(parityAlert).toBeDefined();
    expect(parityAlert.severity).toBe("warning");
  });
});

describe("cockpit route — broker heartbeat projection", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("projects per-broker heartbeat age into runtime rows", async () => {
    const tenSecondsAgo = new Date(Date.now() - 10_000).toISOString();
    mockAllEndpoints(
      makeStatusData({
        broker_heartbeats: {
          alpaca: { last_success_ts: tenSecondsAgo, healthy: true },
          ibkr: { last_success_ts: tenSecondsAgo, healthy: false },
        },
      }),
      {
        portfolioSources: [
          { id: "alpaca-src", broker_type: "alpaca" },
          { id: "ibkr-src", broker_type: "ibkr" },
        ],
      },
    );

    const response = await GET(makeRequest("broker-heartbeat-row"));
    const body = await response.json();
    const alpaca = body.status.brokers.find((row: { broker: string }) => row.broker === "alpaca");
    expect(alpaca).toBeDefined();
    expect(typeof alpaca.heartbeat_ts === "string" || alpaca.heartbeat_ts === null).toBe(true);
    expect(typeof alpaca.stale_age_seconds === "number" || alpaca.stale_age_seconds === null).toBe(true);
  });
});
