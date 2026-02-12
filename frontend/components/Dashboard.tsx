'use client';

import { useMetrics } from '@/lib/api';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Activity, DollarSign, BarChart3, LogOut, Wifi, WifiOff } from 'lucide-react';

export default function Dashboard() {
    const { metrics, isLoading, isError } = useMetrics();
    const router = useRouter();

    const handleLogout = () => {
        document.cookie = 'token=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT';
        router.push('/login');
    };

    const isOffline = isError || (metrics && !metrics.status);

    return (
        <div className="min-h-screen bg-slate-50 dark:bg-slate-900 p-8">
            <div className="max-w-7xl mx-auto space-y-8">
                <header className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-slate-100">Apex Dashboard</h1>
                        <p className="text-slate-500 dark:text-slate-400">Real-time execution monitoring.</p>
                    </div>
                    <Button variant="outline" onClick={handleLogout}>
                        <LogOut className="mr-2 h-4 w-4" />
                        Logout
                    </Button>
                </header>

                {isOffline && (
                    <Alert variant="destructive">
                        <WifiOff className="h-4 w-4" />
                        <AlertTitle>Disconnected</AlertTitle>
                        <AlertDescription>
                            Backend service is unreachable. Displaying cached data if available.
                        </AlertDescription>
                    </Alert>
                )}

                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Status</CardTitle>
                            {isOffline ? <WifiOff className="h-4 w-4 text-red-500" /> : <Wifi className="h-4 w-4 text-green-500" />}
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">{isOffline ? 'Offline' : 'Active'}</div>
                            <p className="text-xs text-muted-foreground">
                                {isOffline ? 'Backend unreachable' : 'Connected to engine'}
                            </p>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Total Trades</CardTitle>
                            <Activity className="h-4 w-4 text-muted-foreground" />
                        </CardHeader>
                        <CardContent>
                            {isLoading ? (
                                <div className="h-8 w-24 bg-slate-200 animate-pulse rounded" />
                            ) : (
                                <>
                                    <div className="text-2xl font-bold">{metrics?.trades_count || 0}</div>
                                    <p className="text-xs text-muted-foreground">Executed orders</p>
                                </>
                            )}
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Slippage (Bps)</CardTitle>
                            <BarChart3 className="h-4 w-4 text-muted-foreground" />
                        </CardHeader>
                        <CardContent>
                            {isLoading ? (
                                <div className="h-8 w-24 bg-slate-200 animate-pulse rounded" />
                            ) : (
                                <>
                                    <div className="text-2xl font-bold">{(metrics?.total_slippage || 0).toFixed(2)}</div>
                                    <p className="text-xs text-muted-foreground">Cumulative basis points</p>
                                </>
                            )}
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Commissions</CardTitle>
                            <DollarSign className="h-4 w-4 text-muted-foreground" />
                        </CardHeader>
                        <CardContent>
                            {isLoading ? (
                                <div className="h-8 w-24 bg-slate-200 animate-pulse rounded" />
                            ) : (
                                <>
                                    <div className="text-2xl font-bold">${(metrics?.total_commission || 0).toFixed(2)}</div>
                                    <p className="text-xs text-muted-foreground">Total fees paid</p>
                                </>
                            )}
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}
