import { expect, test } from "@playwright/test";

test.describe.configure({ mode: "serial" });
test.setTimeout(60_000);

const dashboardMetrics = {
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

const dashboardCockpit = {
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

const plansPayload = [
  {
    code: "free",
    name: "Free",
    tier: "free",
    monthly_usd: 0,
    annual_usd: 0,
    recommended: false,
    target_user: "Personal",
    usp: "Starter access",
    feature_highlights: ["Basic dashboard", "Paper trading", "Delayed analytics"],
    feature_limits: { orders: 25 },
  },
  {
    code: "pro",
    name: "Pro",
    tier: "pro",
    monthly_usd: 149,
    annual_usd: 1490,
    recommended: true,
    target_user: "Active desk",
    usp: "Production controls",
    feature_highlights: ["Live controls", "Risk reports", "Attribution"],
    feature_limits: { orders: -1 },
  },
];

type ViewPreset = {
  name: string;
  viewport: { width: number; height: number };
};

const presets: ViewPreset[] = [
  { name: "desktop", viewport: { width: 1440, height: 900 } },
  { name: "mobile", viewport: { width: 390, height: 844 } },
];

function isIgnorableDevChunkError(message: string): boolean {
  const lower = message.toLowerCase();
  if (lower.includes("failed to load chunk") && lower.includes("hmr-client")) {
    return true;
  }
  if (lower.includes("due to access control checks")) {
    return true;
  }
  if (lower.includes("__nextjs_original-stack-frames")) {
    return true;
  }
  return false;
}

for (const preset of presets) {
  test(`manual QA sweep (${preset.name})`, async ({ browser }) => {
    const context = await browser.newContext({ viewport: preset.viewport });
    const page = await context.newPage();
    const consoleErrors: string[] = [];

    page.on("console", (msg) => {
      if (msg.type() === "error") {
        const text = msg.text();
        if (!isIgnorableDevChunkError(text)) {
          consoleErrors.push(text);
        }
      }
    });
    page.on("pageerror", (error) => {
      if (!isIgnorableDevChunkError(error.message)) {
        consoleErrors.push(error.message);
      }
    });

    await page.route("**/api/v1/plans", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(plansPayload),
      });
    });
    await page.route("**/api/v1/metrics", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(dashboardMetrics),
      });
    });
    await page.route("**/api/v1/cockpit", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(dashboardCockpit),
      });
    });
    await page.route("**/api/v1/portfolio/balance", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ total_equity: 105000, breakdown: [{ source: "ibkr-paper" }] }),
      });
    });
    await page.route("**/api/v1/portfolio/sources", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([]),
      });
    });

    await page.goto("/login", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("button", { name: "Authenticate" })).toBeVisible({ timeout: 15000 });

    await page.goto("/pricing", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: /Hedge-fund-grade automation/i })).toBeVisible({ timeout: 15000 });

    await page.goto("/settings", { waitUntil: "domcontentloaded" });
    await expect(page.getByText("You must be signed in to view settings.")).toBeVisible({ timeout: 15000 });

    await context.addCookies([{ name: "token", value: "mock-token", domain: "localhost", path: "/" }]);

    await page.goto("/dashboard", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: "Apex Dashboard" })).toBeVisible({ timeout: 15000 });

    for (const route of ["/login", "/pricing", "/settings", "/dashboard"]) {
      await page.goto(route, { waitUntil: "domcontentloaded" });
      if (route === "/dashboard") {
        await expect(page.getByRole("heading", { name: "Apex Dashboard" })).toBeVisible({ timeout: 15000 });
      }
      const hasHorizontalOverflow = await page.evaluate(() => {
        return document.documentElement.scrollWidth > window.innerWidth + 1;
      });
      expect(hasHorizontalOverflow, `${preset.name} overflow on ${route}`).toBeFalsy();
    }

    await expect
      .soft(consoleErrors, `${preset.name} console/page errors`)
      .toEqual([]);

    await context.close();
  });
}
