import { test, expect } from '@playwright/test';

test.describe('Dashboard Resilience', () => {
    // Mock login before each test
    test.beforeEach(async ({ page }) => {
        // Set a mock token
        await page.context().addCookies([
            { name: 'token', value: 'mock-token', domain: 'localhost', path: '/' }
        ]);
    });

    test('should load dashboard and display metrics', async ({ page }) => {
        // Mock the metrics API response
        await page.route('/api/v1/metrics', async route => {
            await route.fulfill({
                json: {
                    status: 'success',
                    trades_count: 42,
                    total_slippage: 12.5,
                    total_commission: 5.0,
                    timestamp: new Date().toISOString()
                }
            });
        });

        await page.goto('/dashboard');

        // Verify metrics are displayed
        await expect(page.getByText('42')).toBeVisible();
        await expect(page.getByText('12.50')).toBeVisible();
        await expect(page.getByText('$5.00')).toBeVisible();

        // Verify status is active
        await expect(page.getByText('Active')).toBeVisible();
    });

    test('should show disconnected state when backend is offline', async ({ page }) => {
        // Mock API failure
        await page.route('/api/v1/metrics', async route => {
            await route.abort('failed');
        });

        await page.goto('/dashboard');

        // Verify offline alert
        await expect(page.getByText('Disconnected')).toBeVisible();
        await expect(page.getByText('Backend service is unreachable')).toBeVisible();

        // Verify status is offline
        await expect(page.getByText('Offline')).toBeVisible();
    });
});
