/**
 * @jest-environment jsdom
 */

import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { jest } from '@jest/globals';
import Dashboard from '../Dashboard';
import { AuthProvider } from '../auth/AuthProvider';

// Mock WebSocket hook
const mockSendMessage = jest.fn();
const mockLastMessage = { data: JSON.stringify({ type: 'connected' }) };

jest.mock('@/hooks/useWebSocket');
const mockUseWebSocket = require('@/hooks/useWebSocket').useWebSocket;

// Mock auth context
jest.mock('@/components/auth/AuthProvider');
const mockUseAuth = require('@/components/auth/AuthProvider').useAuthContext;

// Mock child components
jest.mock('../PerformanceCharts', () => {
  return function MockPerformanceCharts() {
    return <div data-testid="performance-charts">Performance Charts</div>;
  };
});

jest.mock('../SectorChart', () => {
  return function MockSectorChart() {
    return <div data-testid="sector-chart">Sector Chart</div>;
  };
});

jest.mock('../AlertNotifications', () => {
  return function MockAlertNotifications({ alerts }: any) {
    return (
      <div data-testid="alert-notifications">
        {alerts?.map((alert: any) => (
          <div key={alert.id}>{alert.title}</div>
        ))}
      </div>
    );
  };
});

jest.mock('../ConnectionStatus', () => {
  return function MockConnectionStatus() {
    return <div data-testid="connection-status">Connected</div>;
  };
});

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
}));

describe('Dashboard Integration Tests', () => {
  const mockTradingState = {
    timestamp: '2024-01-01T12:00:00Z',
    capital: 1000000,
    initial_capital: 1000000,
    positions: {
      'AAPL': {
        qty: 100,
        side: 'long',
        avg_price: 150.0,
        current_price: 155.0,
        pnl: 500,
        pnl_pct: 0.033,
        signal_direction: 'long'
      }
    },
    daily_pnl: 500,
    total_pnl: 5000,
    sector_exposure: {
      'Technology': 0.6,
      'Healthcare': 0.4
    },
    open_positions: 1,
    total_trades: 10,
    sharpe_ratio: 1.5,
    win_rate: 0.6,
    max_drawdown: -0.05,
    alerts: [],
    equity_history: [
      {
        timestamp: '2024-01-01T10:00:00Z',
        equity: 1000000,
        drawdown: 0,
        sharpe: 1.5
      }
    ]
  };

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock WebSocket hook
    mockUseWebSocket.mockReturnValue({
      lastMessage: mockLastMessage,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    // Mock auth context
    mockUseAuth.mockReturnValue({
      user: { id: 'test-user', email: 'test@example.com' },
      token: 'mock-token',
      login: jest.fn(),
      logout: jest.fn(),
      loading: false
    });
  });

  test('renders dashboard with all components', () => {
    render(
      <AuthProvider>
        <Dashboard />
      </AuthProvider>
    );
    
    expect(screen.getByText('APEX Trading System')).toBeInTheDocument();
    expect(screen.getByText('Live Trading Dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('connection-status')).toBeInTheDocument();
  });

  test('handles WebSocket connection state', async () => {
    // Simulate connection
    mockUseWebSocket.mockReturnValue({
      lastMessage: { data: JSON.stringify({ type: 'connected' }) },
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('connection-status')).toBeInTheDocument();
    });
  });

  test('handles WebSocket disconnection', async () => {
    // Simulate disconnection
    mockUseWebSocket.mockReturnValue({
      lastMessage: null,
      sendMessage: mockSendMessage,
      readyState: 3 // CLOSED
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/disconnected/i)).toBeInTheDocument();
    });
  });

  test('displays trading data when received', async () => {
    // Simulate trading state message
    const tradingStateMessage = {
      data: JSON.stringify({
        type: 'trading_state',
        data: mockTradingState
      })
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: tradingStateMessage,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('$1,000,000')).toBeInTheDocument();
      expect(screen.getByText('$500')).toBeInTheDocument();
      expect(screen.getByText('1')).toBeInTheDocument();
    });
  });

  test('handles alerts correctly', async () => {
    const stateWithAlerts = {
      ...mockTradingState,
      alerts: [
        {
          id: 'alert-1',
          severity: 'warning' as const,
          title: 'High Volatility',
          message: 'Market volatility is elevated',
          timestamp: '2024-01-01T12:00:00Z',
          source: 'system'
        }
      ]
    };

    const alertMessage = {
      data: JSON.stringify({
        type: 'trading_state',
        data: stateWithAlerts
      })
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: alertMessage,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('High Volatility')).toBeInTheDocument();
      expect(screen.getByTestId('alert-notifications')).toBeInTheDocument();
    });
  });

  test('handles circuit breaker activation', async () => {
    const stateWithCircuitBreaker = {
      ...mockTradingState,
      circuit_breaker_active: true,
      circuit_breaker_reason: 'High volatility detected'
    };

    const circuitBreakerMessage = {
      data: JSON.stringify({
        type: 'trading_state',
        data: stateWithCircuitBreaker
      })
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: circuitBreakerMessage,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/circuit breaker/i)).toBeInTheDocument();
      expect(screen.getByText('High volatility detected')).toBeInTheDocument();
    });
  });

  test('switches between tabs correctly', async () => {
    render(<Dashboard />);
    
    // Find and click charts tab
    const chartsTab = screen.getByText('Charts');
    fireEvent.click(chartsTab);

    await waitFor(() => {
      expect(screen.getByTestId('performance-charts')).toBeInTheDocument();
    });
  });

  test('handles real-time updates', async () => {
    render(<Dashboard />);

    // Initial state
    const initialMessage = {
      data: JSON.stringify({
        type: 'trading_state',
        data: mockTradingState
      })
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: initialMessage,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    const { rerender } = render(<Dashboard />);

    // Simulate real-time update
    const updatedState = {
      ...mockTradingState,
      capital: 1001000,
      daily_pnl: 1000
    };

    const updatedMessage = {
      data: JSON.stringify({
        type: 'trading_state',
        data: updatedState
      })
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: updatedMessage,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    rerender(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('$1,001,000')).toBeInTheDocument();
      expect(screen.getByText('$1,000')).toBeInTheDocument();
    });
  });

  test('handles authentication state', async () => {
    // Mock unauthenticated state
    mockUseAuth.mockReturnValue({
      user: null,
      token: null,
      login: jest.fn(),
      logout: jest.fn(),
      loading: false
    });

    render(<Dashboard />);

    await waitFor(() => {
      // Should show login or redirect
      expect(screen.getByText(/login/i)).toBeInTheDocument();
    });
  });

  test('handles loading state', async () => {
    // Mock loading state
    mockUseAuth.mockReturnValue({
      user: null,
      token: null,
      login: jest.fn(),
      logout: jest.fn(),
      loading: true
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });

  test('displays sector exposure chart', async () => {
    const sectorMessage = {
      data: JSON.stringify({
        type: 'trading_state',
        data: mockTradingState
      })
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: sectorMessage,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('sector-chart')).toBeInTheDocument();
    });
  });

  test('handles error states gracefully', async () => {
    // Mock error state
    mockUseWebSocket.mockReturnValue({
      lastMessage: {
        data: JSON.stringify({
          type: 'error',
          message: 'Connection failed'
        })
      },
      sendMessage: mockSendMessage,
      readyState: 3
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
      expect(screen.getByText(/connection failed/i)).toBeInTheDocument();
    });
  });

  test('formats large numbers correctly', async () => {
    const largeNumbersState = {
      ...mockTradingState,
      capital: 10000000,
      total_pnl: 500000,
      daily_pnl: 25000
    };

    const largeNumbersMessage = {
      data: JSON.stringify({
        type: 'trading_state',
        data: largeNumbersState
      })
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: largeNumbersMessage,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('$10,000,000')).toBeInTheDocument();
      expect(screen.getByText('$500,000')).toBeInTheDocument();
      expect(screen.getByText('$25,000')).toBeInTheDocument();
    });
  });

  test('handles empty trading state', async () => {
    const emptyState = {
      ...mockTradingState,
      positions: {},
      open_positions: 0,
      total_trades: 0
    };

    const emptyStateMessage = {
      data: JSON.stringify({
        type: 'trading_state',
        data: emptyState
      })
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: emptyStateMessage,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('0')).toBeInTheDocument(); // Open positions
      expect(screen.getByText('0')).toBeInTheDocument(); // Total trades
    });
  });
});
