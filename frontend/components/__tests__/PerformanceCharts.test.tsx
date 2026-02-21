/**
 * @jest-environment jsdom
 */

import type { ReactNode } from "react";
import { render, screen } from "@testing-library/react";
import PerformanceCharts from "../PerformanceCharts";

jest.mock("recharts", () => ({
  LineChart: () => <div data-testid="line-chart" />,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ResponsiveContainer: ({ children }: { children?: ReactNode }) => <div data-testid="responsive-container">{children}</div>,
  AreaChart: () => <div data-testid="area-chart" />,
  Area: () => <div data-testid="area" />,
  ReferenceLine: () => <div data-testid="reference-line" />,
}));

jest.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: { children?: ReactNode; [key: string]: unknown }) => (
      <div {...props}>{children}</div>
    ),
  },
}));

describe("PerformanceCharts", () => {
  const equityHistory = [
    { timestamp: "2026-02-21T10:00:00Z", equity: 1000000, drawdown: 0, sharpe: 1.5 },
    { timestamp: "2026-02-21T11:00:00Z", equity: 1010000, drawdown: -0.01, sharpe: 1.7 },
  ];

  test("shows waiting state with insufficient history", () => {
    render(<PerformanceCharts equityHistory={[]} initialCapital={1000000} />);
    expect(screen.getByText("Waiting for equity history data...")).toBeInTheDocument();
  });

  test("renders chart sections when history is available", () => {
    render(<PerformanceCharts equityHistory={equityHistory} initialCapital={1000000} />);
    expect(screen.getByText("Equity Curve")).toBeInTheDocument();
    expect(screen.getByText("Drawdown")).toBeInTheDocument();
    expect(screen.getByText("Rolling Sharpe Ratio")).toBeInTheDocument();
    expect(screen.getAllByTestId("area-chart").length).toBeGreaterThanOrEqual(1);
  });

  test("shows current pnl and sharpe indicators", () => {
    render(<PerformanceCharts equityHistory={equityHistory} initialCapital={1000000} />);
    expect(screen.getByText("+1.00%")).toBeInTheDocument();
    expect(screen.getByText("Current: 1.70")).toBeInTheDocument();
  });
});
