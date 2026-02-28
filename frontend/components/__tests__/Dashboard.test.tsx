/**
 * @jest-environment jsdom
 */

import { fireEvent, render, screen } from "@testing-library/react";
import Dashboard from "../Dashboard";
import { useCockpitData, useMetrics } from "@/lib/api";
import { useAuthContext } from "@/components/auth/AuthProvider";
import { useTheme } from "@/components/theme/ThemeProvider";
import { useWebSocket } from "@/hooks/useWebSocket";

const pushMock = jest.fn();
const replaceMock = jest.fn();
const logoutMock = jest.fn();
const fetchMock = jest.fn();

jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
    replace: replaceMock,
  }),
}));

jest.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: jest.fn(),
}));

jest.mock("@/lib/api", () => ({
  useMetrics: jest.fn(),
  useCockpitData: jest.fn(),
}));

jest.mock("@/components/theme/ThemeProvider", () => ({
  useTheme: jest.fn(),
}));

jest.mock("@/components/auth/AuthProvider", () => ({
  useAuthContext: jest.fn(),
}));

const mockMetrics = {
  status: true,
  timestamp: "2026-02-21T12:00:00Z",
  capital: 105000,
  starting_capital: 100000,
  daily_pnl: 1200,
  total_pnl: 5000,
  max_drawdown: -0.04,
  sharpe_ratio: 1.35,
  win_rate: 0.58,
  open_positions: 3,
  trades_count: 42,
};

const mockCockpit = {
  status: {
    online: true,
    api_reachable: true,
    state_fresh: true,
    timestamp: "2026-02-21T12:00:00Z",
    capital: 105000,
    starting_capital: 100000,
    daily_pnl: 1200,
    total_pnl: 5000,
    max_drawdown: -0.04,
    sharpe_ratio: 1.35,
    win_rate: 0.58,
    open_positions: 3,
    option_positions: 0,
    open_positions_total: 3,
    total_trades: 42,
  },
  positions: [],
  derivatives: [],
  attribution: {
    closed_trades: 42,
    open_positions_tracked: 3,
    gross_pnl: 5600,
    net_pnl: 5000,
    commissions: 300,
    modeled_execution_drag: 180,
    modeled_slippage_drag: 120,
    sleeves: [],
  },
  usp: {
    engine: "online",
    score: 76,
    band: "improving" as const,
    sharpe_progress_pct: 90,
    drawdown_budget_used_pct: 26,
    alpha_retention_pct: 84,
    execution_drag_pct_of_gross: 5,
  },
  social_audit: {
    available: true,
    unauthorized: false,
    transport_error: false,
    status_code: 200,
    warning: null,
    count: 0,
    events: [],
  },
  alerts: [],
  notes: [],
};

const mockCockpitWithPositions = {
  ...mockCockpit,
  status: {
    ...mockCockpit.status,
    open_positions: 1,
    open_positions_total: 1,
  },
  positions: [
    {
      symbol: "AAPL",
      qty: 10,
      side: "LONG",
      entry: 190,
      current: 194,
      pnl: 40,
      pnl_pct: 2.1,
      signal: 0.4,
      signal_direction: "BUY",
      source_id: "ibkr-1",
      broker_type: "ibkr",
      stale: false,
      source_status: "live",
    },
  ],
};

describe("Dashboard", () => {
  const mockUseWebSocket = useWebSocket as jest.MockedFunction<typeof useWebSocket>;
  const mockUseMetrics = useMetrics as jest.MockedFunction<typeof useMetrics>;
  const mockUseCockpitData = useCockpitData as jest.MockedFunction<typeof useCockpitData>;
  const mockUseTheme = useTheme as jest.MockedFunction<typeof useTheme>;
  const mockUseAuth = useAuthContext as jest.MockedFunction<typeof useAuthContext>;

  beforeEach(() => {
    jest.clearAllMocks();
    global.fetch = fetchMock as unknown as typeof fetch;
    fetchMock.mockImplementation(async () => {
      return {
        ok: false,
        status: 503,
        json: async () => ({}),
      } as Response;
    });

    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      isConnecting: false,
      reconnectAttempt: 0,
      lastError: undefined,
      lastMessage: null,
      retry: jest.fn(),
    });

    mockUseMetrics.mockReturnValue({
      metrics: mockMetrics,
      isLoading: false,
      isError: false,
      error: undefined,
    });

    mockUseCockpitData.mockReturnValue({
      data: mockCockpit,
      isLoading: false,
      isError: false,
      error: undefined,
    });

    mockUseTheme.mockReturnValue({
      theme: "dark",
      setTheme: jest.fn(),
      toggleTheme: jest.fn(),
    });

    mockUseAuth.mockReturnValue({
      user: null,
      accessToken: "token",
      isLoading: false,
      isAuthenticated: true,
      login: async () => ({ ok: true }),
      register: async () => ({ ok: true }),
      logout: logoutMock,
      refreshUser: async () => { },
      getToken: () => "token",
    });
  });

  test("renders dashboard shell with live heading", () => {
    render(<Dashboard />);
    expect(screen.getByRole("heading", { name: "Apex Terminal" })).toBeInTheDocument();
    expect(screen.getByText("Function Readiness")).toBeInTheDocument();
  });

  test("logs out and routes to login", () => {
    render(<Dashboard />);
    fireEvent.click(screen.getByRole("button", { name: "Logout" }));
    expect(logoutMock).toHaveBeenCalledTimes(1);
    expect(pushMock).toHaveBeenCalledWith("/login");
  });

  test("sanitizes extreme metric outliers before rendering KPI cards", () => {
    mockUseMetrics.mockReturnValue({
      metrics: {
        ...mockMetrics,
        sharpe_ratio: Number("-92962852034076208"),
        win_rate: 58,
        max_drawdown: -9999,
      },
      isLoading: false,
      isError: false,
      error: undefined,
    });

    render(<Dashboard />);
    expect(screen.queryByText("-92962852034076208.00")).not.toBeInTheDocument();
    expect(screen.queryByText(/-9999/)).not.toBeInTheDocument();
    expect(screen.getAllByText("1.35").length).toBeGreaterThan(0);
  });

  test("keeps position rows visible when websocket payload is temporarily empty", () => {
    mockUseMetrics.mockReturnValue({
      metrics: { ...mockMetrics, open_positions: 1 },
      isLoading: false,
      isError: false,
      error: undefined,
    });
    mockUseCockpitData.mockReturnValue({
      data: mockCockpitWithPositions,
      isLoading: false,
      isError: false,
      error: undefined,
    });
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      isConnecting: false,
      reconnectAttempt: 0,
      lastError: undefined,
      lastMessage: {
        type: "state_update",
        timestamp: "2026-02-21T12:00:10Z",
        open_positions: 1,
        positions: {},
      },
      retry: jest.fn(),
    });

    render(<Dashboard />);

    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.queryByText("No positions in API state.")).not.toBeInTheDocument();
  });

  test("retains previous position snapshot during transient cockpit fetch failure", () => {
    mockUseMetrics.mockReturnValue({
      metrics: { ...mockMetrics, open_positions: 1 },
      isLoading: false,
      isError: false,
      error: undefined,
    });
    mockUseCockpitData
      .mockReturnValueOnce({
        data: mockCockpitWithPositions,
        isLoading: false,
        isError: false,
        error: undefined,
      })
      .mockReturnValueOnce({
        data: undefined,
        isLoading: false,
        isError: true,
        error: new Error("transient 500"),
      });
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      isConnecting: false,
      reconnectAttempt: 0,
      lastError: undefined,
      lastMessage: {
        type: "state_update",
        timestamp: "2026-02-21T12:00:10Z",
        open_positions: 0,
        positions: {},
      },
      retry: jest.fn(),
    });

    const { rerender } = render(<Dashboard />);
    expect(screen.getByText("AAPL")).toBeInTheDocument();

    rerender(<Dashboard />);
    expect(screen.getByText("AAPL")).toBeInTheDocument();
  });

  test("does not let zeroed websocket metrics overwrite non-zero cockpit KPIs", () => {
    mockUseMetrics.mockReturnValue({
      metrics: mockMetrics,
      isLoading: false,
      isError: false,
      error: undefined,
    });
    mockUseCockpitData.mockReturnValue({
      data: mockCockpit,
      isLoading: false,
      isError: false,
      error: undefined,
    });
    mockUseWebSocket.mockReturnValue({
      isConnected: true,
      isConnecting: false,
      reconnectAttempt: 0,
      lastError: undefined,
      lastMessage: {
        type: "state_update",
        timestamp: "2026-02-21T12:00:10Z",
        capital: 105000,
        daily_pnl: 0,
        total_pnl: 0,
        sharpe_ratio: 0,
        win_rate: 0,
        open_positions: 0,
        total_trades: 0,
        positions: {},
      },
      retry: jest.fn(),
    });

    render(<Dashboard />);

    // Sharpe should still render from cockpit/REST baseline instead of websocket zeros.
    expect(screen.getAllByText("1.35").length).toBeGreaterThan(0);
  });

  test("renders broker heartbeat stale-age details in Function Readiness", () => {
    mockUseCockpitData.mockReturnValue({
      data: {
        ...mockCockpit,
        status: {
          ...mockCockpit.status,
          brokers: [
            {
              broker: "alpaca",
              status: "live",
              mode: "trading",
              configured: true,
              stale: false,
              source_count: 1,
              live_source_count: 1,
              source_ids: ["alpaca-1"],
              total_equity: 100000,
              heartbeat_ts: "2026-02-21T12:00:00Z",
              stale_age_seconds: 7,
            },
            {
              broker: "ibkr",
              status: "stale",
              mode: "idle",
              configured: true,
              stale: true,
              source_count: 1,
              live_source_count: 0,
              source_ids: ["ibkr-1"],
              total_equity: 900000,
              heartbeat_ts: "2026-02-21T11:59:00Z",
              stale_age_seconds: 45,
            },
          ],
          active_broker: "alpaca",
        },
      },
      isLoading: false,
      isError: false,
      error: undefined,
    });

    render(<Dashboard />);

    expect(screen.getByText(/trading active .* hb 7s @/i)).toBeInTheDocument();
    expect(screen.getByText(/idle \(positions\/metrics\) .* hb 45s @/i)).toBeInTheDocument();
  });
});
