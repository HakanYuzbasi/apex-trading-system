/**
 * @jest-environment jsdom
 */

import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { jest } from '@jest/globals';
import PerformanceCharts from '../PerformanceCharts';

// Mock recharts
jest.mock('recharts', () => ({
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  AreaChart: ({ children }: any) => <div data-testid="area-chart">{children}</div>,
  Area: () => <div data-testid="area" />,
  ReferenceLine: () => <div data-testid="reference-line" />,
}));

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
}));

describe('PerformanceCharts Component', () => {
  const mockEquityData = [
    { timestamp: '2024-01-01T10:00:00Z', equity: 1000000, drawdown: 0, sharpe: 1.5 },
    { timestamp: '2024-01-01T11:00:00Z', equity: 1005000, drawdown: -0.01, sharpe: 1.6 },
    { timestamp: '2024-01-01T12:00:00Z', equity: 1010000, drawdown: -0.005, sharpe: 1.7 },
  ];

  const mockInitialCapital = 1000000;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders performance charts container', () => {
    render(
      <PerformanceCharts 
        equityHistory={mockEquityData}
        initialCapital={mockInitialCapital}
      />
    );

    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
  });

  test('displays equity curve chart', () => {
    render(
      <PerformanceCharts 
        equityHistory={mockEquityData}
        initialCapital={mockInitialCapital}
      />
    );

    expect(screen.getByTestId('area-chart')).toBeInTheDocument();
  });

  test('handles empty data gracefully', () => {
    render(
      <PerformanceCharts 
        equityHistory={[]}
        initialCapital={mockInitialCapital}
      />
    );

    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
  });

  test('displays drawdown chart', () => {
    render(
      <PerformanceCharts 
        equityHistory={mockEquityData}
        initialCapital={mockInitialCapital}
      />
    );

    expect(screen.getByTestId('area-chart')).toBeInTheDocument();
  });

  test('formats currency values correctly', () => {
    render(
      <PerformanceCharts 
        equityHistory={mockEquityData}
        initialCapital={mockInitialCapital}
      />
    );

    // Check if equity values are displayed
    expect(screen.getByText('$1,000,000')).toBeInTheDocument();
    expect(screen.getByText('$1,010,000')).toBeInTheDocument();
  });

  test('formats percentage values correctly', () => {
    render(
      <PerformanceCharts 
        equityHistory={mockEquityData}
        initialCapital={mockInitialCapital}
      />
    );

    // Check if drawdown percentages are displayed
    expect(screen.getByText('0.00%')).toBeInTheDocument();
    expect(screen.getByText('-1.00%')).toBeInTheDocument();
  });

  test('displays Sharpe ratio', () => {
    render(
      <PerformanceCharts 
        equityHistory={mockEquityData}
        initialCapital={mockInitialCapital}
      />
    );

    expect(screen.getByText('1.50')).toBeInTheDocument();
  });

  test('handles chart interactions', async () => {
    render(
      <PerformanceCharts 
        equityHistory={mockEquityData}
        initialCapital={mockInitialCapital}
      />
    );

    const chart = screen.getByTestId('area-chart');
    
    // Test that chart renders
    expect(chart).toBeInTheDocument();
  });

  test('displays multiple data points correctly', () => {
    const extendedData = [
      ...mockEquityData,
      { timestamp: '2024-01-01T13:00:00Z', equity: 1015000, drawdown: -0.003, sharpe: 1.8 },
      { timestamp: '2024-01-01T14:00:00Z', equity: 1020000, drawdown: -0.001, sharpe: 1.9 },
    ];

    render(
      <PerformanceCharts 
        equityHistory={extendedData}
        initialCapital={mockInitialCapital}
      />
    );

    expect(screen.getByText('$1,020,000')).toBeInTheDocument();
  });

  test('handles negative equity correctly', () => {
    const negativeEquityData = [
      { timestamp: '2024-01-01T10:00:00Z', equity: 1000000, drawdown: 0, sharpe: 1.5 },
      { timestamp: '2024-01-01T11:00:00Z', equity: 950000, drawdown: -0.05, sharpe: 1.2 },
    ];

    render(
      <PerformanceCharts 
        equityHistory={negativeEquityData}
        initialCapital={mockInitialCapital}
      />
    );

    expect(screen.getByText('$950,000')).toBeInTheDocument();
    expect(screen.getByText('-5.00%')).toBeInTheDocument();
  });

  test('displays zero values correctly', () => {
    const zeroData = [
      { timestamp: '2024-01-01T10:00:00Z', equity: 1000000, drawdown: 0, sharpe: 0 },
    ];

    render(
      <PerformanceCharts 
        equityHistory={zeroData}
        initialCapital={mockInitialCapital}
      />
    );

    expect(screen.getByText('$1,000,000')).toBeInTheDocument();
    expect(screen.getByText('0.00')).toBeInTheDocument();
  });

  test('handles large numbers correctly', () => {
    const largeData = [
      { timestamp: '2024-01-01T10:00:00Z', equity: 10000000, drawdown: 0, sharpe: 2.5 },
    ];

    render(
      <PerformanceCharts 
        equityHistory={largeData}
        initialCapital={mockInitialCapital}
      />
    );

    expect(screen.getByText('$10,000,000')).toBeInTheDocument();
  });

  test('handles decimal precision correctly', () => {
    const preciseData = [
      { timestamp: '2024-01-01T10:00:00Z', equity: 1000000.55, drawdown: -0.00123, sharpe: 1.567 },
    ];

    render(
      <PerformanceCharts 
        equityHistory={preciseData}
        initialCapital={mockInitialCapital}
      />
    );

    expect(screen.getByText('$1,000,001')).toBeInTheDocument();
    expect(screen.getByText('-0.12%')).toBeInTheDocument();
  });
});
