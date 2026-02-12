import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
    test('should redirect to login when accessing dashboard unauthenticated', async ({ page }) => {
        await page.goto('/dashboard');
        await expect(page).toHaveURL(/.*\/login/);
    });

    test('should login successfully with valid credentials', async ({ page }) => {
        await page.goto('/login');

        // Fill credentials
        await page.fill('input[type="text"]', 'admin'); // Our simplified login just needs any user
        await page.fill('input[type="password"]', 'password');

        // Click login
        await page.click('button[type="submit"]');

        // Should redirect to dashboard
        await expect(page).toHaveURL(/.*\/dashboard/);

        // Dashboard should be visible
        await expect(page.getByText('Apex Dashboard')).toBeVisible();
        await expect(page.getByText('Real-time execution monitoring')).toBeVisible();
    });

    test('should logout successfully', async ({ page }) => {
        // Login first
        await page.goto('/login');
        await page.fill('input[type="text"]', 'admin');
        await page.fill('input[type="password"]', 'password');
        await page.click('button[type="submit"]');
        await expect(page).toHaveURL(/.*\/dashboard/);

        // Logout
        await page.click('button:has-text("Logout")');

        // Should redirect to login
        await expect(page).toHaveURL(/.*\/login/);
    });
});
