import { expect, test } from "@playwright/test";

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
    band: "improving",
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

test.describe("Session Re-authentication", () => {
  test("session_expired banner -> re-authenticate -> return to dashboard", async ({ page }) => {
    await page.route("**/api/auth/login", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          access_token: "test-access-token",
          refresh_token: "test-refresh-token",
          token_type: "bearer",
        }),
      });
    });

    await page.route("**/api/auth/me", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          user_id: "u-admin",
          username: "admin",
          email: "admin@example.com",
          roles: ["admin"],
          tier: "enterprise",
        }),
      });
    });

    await page.route("**/api/v1/metrics", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockMetrics),
      });
    });

    await page.route("**/api/v1/cockpit", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockCockpit),
      });
    });

    await page.route("**/api/v1/portfolio/balance", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          total_equity: 105000,
          breakdown: [{ source: "ibkr-paper" }],
        }),
      });
    });

    await page.route("**/api/v1/portfolio/sources", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([]),
      });
    });

    await page.goto("/login?reason=session_expired&returnUrl=%2Fdashboard");
    await expect(page.getByText("Session expired. Please authenticate again.")).toBeVisible();

    await page.getByLabel("Username").fill("admin");
    await page.getByLabel("Password / Master Key").fill("P3rO_f73zKfHHkt2WfxJ7zDZ");
    await page.getByRole("button", { name: "Authenticate" }).click();

    await expect(page).toHaveURL(/\/dashboard$/);
    await expect(page.getByRole("heading", { name: "Apex Dashboard" })).toBeVisible();
  });
});
