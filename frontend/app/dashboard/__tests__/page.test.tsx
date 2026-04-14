/**
 * @jest-environment jsdom
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import DashboardOverview from "../page";
import { changeBrokerMode, useBrokerMode, useCockpitData, useSessionMetrics } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";

jest.mock("next/link", () => ({
  __esModule: true,
  default: ({ children, href, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

jest.mock("@/lib/api", () => ({
  useSessionMetrics: jest.fn(),
  useCockpitData: jest.fn(),
  useBrokerMode: jest.fn(),
  changeBrokerMode: jest.fn(),
}));

jest.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: jest.fn(),
}));

jest.mock("@/components/dashboard/AdvancedMetricsPanel", () => ({
  __esModule: true,
  default: () => <div data-testid="advanced-metrics-panel">advanced metrics mock</div>,
}));

jest.mock("@/components/dashboard/PitchMetricsRibbon", () => ({
  __esModule: true,
  default: () => <div data-testid="pitch-metrics-ribbon">pitch metrics mock</div>,
}));

jest.mock("@/components/dashboard/ShadowTerminal", () => ({
  __esModule: true,
  default: () => <div data-testid="shadow-terminal">shadow terminal mock</div>,
}));

jest.mock("@/components/ErrorBoundary", () => ({
  ErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe("Dashboard overview page", () => {
  const mockUseSessionMetrics = useSessionMetrics as jest.MockedFunction<typeof useSessionMetrics>;
  const mockUseCockpitData = useCockpitData as jest.MockedFunction<typeof useCockpitData>;
  const mockUseBrokerMode = useBrokerMode as jest.MockedFunction<typeof useBrokerMode>;
  const mockChangeBrokerMode = changeBrokerMode as jest.MockedFunction<typeof changeBrokerMode>;
  const mockUseWebSocket = useWebSocket as jest.MockedFunction<typeof useWebSocket>;
  const mutate = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mutate.mockResolvedValue(undefined);
    mockChangeBrokerMode.mockResolvedValue({ status: "ok", broker_mode: "alpaca" });
    mockUseBrokerMode.mockReturnValue({
      data: { broker_mode: "both" },
      mutate,
    } as never);
    mockUseWebSocket.mockReturnValue({ isConnected: true } as never);
    mockUseCockpitData.mockReturnValue({
      data: {
        status: {
          api_reachable: true,
          total_equity: 123456,
          capital: 999,
          daily_pnl: 250,
          total_pnl: 500,
          open_positions: 4,
          sharpe_ratio: 1.2,
          win_rate: 0.62,
          total_trades: 11,
          brokers: [{ broker: "alpaca", mode: "trading" }],
        },
        alerts: [],
      },
      isLoading: false,
    } as never);
    mockUseSessionMetrics.mockImplementation((sessionType: string) =>
      ({
        data:
          sessionType === "crypto"
            ? {
                capital: 50000,
                sharpe_ratio: -0.4,
                daily_pnl: -1250,
                win_rate: 0.41,
                open_positions: 2,
              }
            : {
                capital: 100000,
                sharpe_ratio: 1.25,
                daily_pnl: 875,
                win_rate: 0.57,
                open_positions: 3,
              },
      }) as never,
    );
  });

  it("renders the institutional risk overlay at the top of the page", () => {
    render(<DashboardOverview />);

    expect(screen.getByTestId("pitch-metrics-ribbon")).toBeInTheDocument();
    expect(screen.getByTestId("shadow-terminal")).toBeInTheDocument();
    expect(screen.getByText("Institutional Risk Overlay")).toBeInTheDocument();
    expect(screen.getByTestId("advanced-metrics-panel")).toBeInTheDocument();
  });

  it("prefers total_equity over capital in the portfolio overview", () => {
    render(<DashboardOverview />);

    expect(screen.getByText("$123,456.00")).toBeInTheDocument();
    expect(screen.queryByText("$999.00")).not.toBeInTheDocument();
  });

  it("renders the negative Sharpe warning state for underperforming sessions", () => {
    render(<DashboardOverview />);

    expect(screen.getByText("Critical: Performance below baseline")).toBeInTheDocument();
  });

  it("renders only one broker mode control and routes changes through the helper", async () => {
    render(<DashboardOverview />);

    expect(screen.getAllByText("Execution Mode")).toHaveLength(1);

    fireEvent.click(screen.getByRole("button", { name: "Full Alpaca" }));

    await waitFor(() => {
      expect(mockChangeBrokerMode).toHaveBeenCalledWith("alpaca");
      expect(mutate).toHaveBeenCalled();
    });
  });
});
