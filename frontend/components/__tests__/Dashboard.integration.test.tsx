/**
 * @jest-environment jsdom
 */

import { render, screen, waitFor } from "@testing-library/react";
import Dashboard from "../Dashboard";

const replaceMock = jest.fn();
const logoutMock = jest.fn();

jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: replaceMock,
  }),
}));

jest.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: () => ({
    isConnected: false,
    isConnecting: false,
    reconnectAttempt: 0,
    lastError: undefined,
    lastMessage: null,
    retry: jest.fn(),
  }),
}));

jest.mock("@/lib/api", () => ({
  useMetrics: () => ({
    metrics: undefined,
    isLoading: false,
    isError: true,
    error: new Error("Session expired"),
  }),
  useCockpitData: () => ({
    data: undefined,
    isLoading: false,
    isError: true,
    error: new Error("Not authenticated"),
  }),
}));

jest.mock("@/components/theme/ThemeProvider", () => ({
  useTheme: () => ({
    theme: "dark",
    setTheme: jest.fn(),
    toggleTheme: jest.fn(),
  }),
}));

jest.mock("@/components/auth/AuthProvider", () => ({
  useAuthContext: () => ({
    user: null,
    accessToken: "token",
    isLoading: false,
    isAuthenticated: true,
    login: jest.fn(),
    register: jest.fn(),
    logout: logoutMock,
    refreshUser: jest.fn(),
    getToken: () => "token",
  }),
}));

describe("Dashboard Session Integration", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("redirects to login when session is expired", async () => {
    render(<Dashboard />);
    expect(screen.getByText("Session expired")).toBeInTheDocument();
    await waitFor(() => {
      expect(logoutMock).toHaveBeenCalledTimes(1);
      expect(replaceMock).toHaveBeenCalledWith("/login?reason=session_expired");
    });
  });
});
