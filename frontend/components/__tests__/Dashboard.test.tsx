/**
 * @jest-environment jsdom
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { jest } from '@jest/globals';
import Dashboard from '../Dashboard';
import * as useWebSocketModule from '@/hooks/useWebSocket';
import * as authModule from '@/components/auth/AuthProvider';

// Mock the WebSocket hook
const mockSendMessage = jest.fn();
const mockLastMessage = { data: JSON.stringify({ type: 'connected' }) };

jest.mock('@/hooks/useWebSocket');
const mockUseWebSocket = useWebSocketModule.useWebSocket as jest.MockedFunction<typeof useWebSocketModule.useWebSocket>;

// Mock the auth context
jest.mock('@/components/auth/AuthProvider');
const mockUseAuth = authModule.useAuthContext as jest.MockedFunction<typeof authModule.useAuthContext>;

// Mock the 3D component
jest.mock('../VolatilitySurface3D', () => {
  return function MockVolatilitySurface3D() {
    return <div data-testid="volatility-surface-3d">3D Surface</div>;
  };
});

describe('Dashboard Component', () => {
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

  test('renders dashboard with initial state', async () => {
    render(<Dashboard />);
    
    // Check for main dashboard elements
    expect(screen.getByText('APEX Trading System')).toBeInTheDocument();
    expect(screen.getByText('Live Trading Dashboard')).toBeInTheDocument();
  });

  test('displays connection status', async () => {
    render(<Dashboard />);
    
    // Connection status component should be present
    await waitFor(() => {
      expect(screen.getByText(/connection/i)).toBeInTheDocument();
    });
  });

  test('handles WebSocket messages correctly', async () => {
    render(<Dashboard />);
    
    // Simulate receiving trading state
    const tradingStateMessage = {
      data: JSON.stringify({
        type: 'trading_state',
        data: mockTradingState
      })
    };

    // Update the mock to return our test message
    mockUseWebSocket.mockReturnValue({
      lastMessage: tradingStateMessage,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    // Re-render with the new message
    render(<Dashboard />);

    // Wait for the component to process the message
    await waitFor(() => {
      expect(screen.getByText('$1,000,000')).toBeInTheDocument();
    });
  });

  test('displays trading metrics when data is available', async () => {
    // Mock WebSocket to return trading state
    mockUseWebSocket.mockReturnValue({
      lastMessage: {
        data: JSON.stringify({
          type: 'trading_state',
          data: mockTradingState
        })
      },
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      // Check for key metrics
      expect(screen.getByText('$1,000,000')).toBeInTheDocument();
      expect(screen.getByText('$500')).toBeInTheDocument(); // Daily P&L
      expect(screen.getByText('1')).toBeInTheDocument(); // Open positions
    });
  });

  test('handles tab navigation', async () => {
    render(<Dashboard />);
    
    // Find and click the charts tab
    const chartsTab = screen.getByText('Charts');
    fireEvent.click(chartsTab);

    await waitFor(() => {
      // Should show charts content
      expect(screen.getByText('Performance Charts')).toBeInTheDocument();
    });
  });

  test('displays alerts when present', async () => {
    const stateWithAlerts = {
      ...mockTradingState,
      alerts: [
        {
          id: 'alert-1',
          severity: 'warning' as const,
          title: 'Test Alert',
          message: 'This is a test alert',
          timestamp: '2024-01-01T12:00:00Z',
          source: 'system'
        }
      ]
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: {
        data: JSON.stringify({
          type: 'trading_state',
          data: stateWithAlerts
        })
      },
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('Test Alert')).toBeInTheDocument();
      expect(screen.getByText('This is a test alert')).toBeInTheDocument();
    });
  });

  test('handles circuit breaker activation', async () => {
    const stateWithCircuitBreaker = {
      ...mockTradingState,
      circuit_breaker_active: true,
      circuit_breaker_reason: 'High volatility detected'
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: {
        data: JSON.stringify({
          type: 'trading_state',
          data: stateWithCircuitBreaker
        })
      },
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText(/circuit breaker/i)).toBeInTheDocument();
      expect(screen.getByText('High volatility detected')).toBeInTheDocument();
    });
  });

  test('displays sector exposure chart', async () => {
    render(<Dashboard />);

    await waitFor(() => {
      // Check for sector chart component
      const sectorChart = screen.getByTestId('sector-chart') || 
                         screen.getByText('Sector Exposure');
      expect(sectorChart).toBeInTheDocument();
    });
  });

  test('handles connection errors gracefully', async () => {
    // Mock WebSocket error state
    mockUseWebSocket.mockReturnValue({
      lastMessage: null,
      sendMessage: mockSendMessage,
      readyState: 3 // CLOSED state
    });

    render(<Dashboard />);

    await waitFor(() => {
      // Should show disconnected state
      expect(screen.getByText(/disconnected/i)).toBeInTheDocument();
    });
  });

  test('formats currency values correctly', async () => {
    mockUseWebSocket.mockReturnValue({
      lastMessage: {
        data: JSON.stringify({
          type: 'trading_state',
          data: mockTradingState
        })
      },
      sendMessage: mockSendMessage,
      readyState: 1
    });

    render(<Dashboard />);

    await waitFor(() => {
      // Check currency formatting
      expect(screen.getByText('$1,000,000')).toBeInTheDocument();
      expect(screen.getByText('$5,000')).toBeInTheDocument(); // Total P&L
    });
  });

  test('updates real-time metrics', async () => {
    mockUseWebSocket.mockReturnValue({
      lastMessage: null,
      sendMessage: mockSendMessage,
      readyState: 1
    });

    const { rerender } = render(<Dashboard />);

    // Simulate real-time updates
    const updatedState = {
      ...mockTradingState,
      capital: 1001000,
      daily_pnl: 1000
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: {
        data: JSON.stringify({
          type: 'trading_state',
          data: updatedState
        })
      },
      sendMessage: mockSendMessage,
      readyState: 1
    });

    rerender(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByText('$1,001,000')).toBeInTheDocument();
      expect(screen.getByText('$1,000')).toBeInTheDocument();
    });
  });
});
