"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Plus, RefreshCw, Server, Trash2, WifiOff, Zap, Timer } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { StatusPip } from "@/components/ui/status-pip";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { cn } from "@/lib/utils";

// ─── Types ─────────────────────────────────────────────────────────────────
type BrokerType = "alpaca" | "ibkr";
type Environment = "paper" | "live";

interface BrokerConnection {
    id: string;
    name: string;
    broker_type: BrokerType;
    environment: Environment;
    is_active: boolean;
    created_at: string;
}

interface PortfolioBalance {
    total_equity: number;
    last_updated: string;
    breakdown: { source: string; equity: number }[];
}

// ─── Sub-components ─────────────────────────────────────────────────────────

function BrokerBadge({ type }: { type: BrokerType }) {
    return (
        <Badge variant={type === "alpaca" ? "alpaca" : "ibkr"} className="border-none">
            {type === "alpaca" ? "Alpaca" : "IBKR"}
        </Badge>
    );
}

function EnvBadge({ env }: { env: Environment }) {
    return (
        <Badge variant={env === "live" ? "warning" : "secondary"} className="border-none">
            {env === "live" ? "Live" : "Paper"}
        </Badge>
    );
}

// ─── Add Connection Modal ────────────────────────────────────────────────────
interface AddConnectionModalProps {
    onClose: () => void;
    onSuccess: () => void;
    accessToken: string | null | undefined;
}

function AddConnectionModal({ onClose, onSuccess, accessToken }: AddConnectionModalProps) {
    const [brokerType, setBrokerType] = useState<BrokerType>("alpaca");
    const [name, setName] = useState("");
    const [env, setEnv] = useState<Environment>("paper");
    const [apiKey, setApiKey] = useState("");
    const [apiSecret, setApiSecret] = useState("");
    const [ibkrHost, setIbkrHost] = useState("");
    const [ibkrPort, setIbkrPort] = useState("7497");
    const [ibkrClientId, setIbkrClientId] = useState("10");

    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState("");

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError("");
        setSubmitting(true);

        try {
            const base = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
            const credentials =
                brokerType === "alpaca"
                    ? { api_key: apiKey, api_secret: apiSecret }
                    : { host: ibkrHost, port: Number(ibkrPort), client_id: Number(ibkrClientId) };

            const res = await fetch(`${base}/brokers/connect`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
                },
                body: JSON.stringify({
                    name,
                    broker_type: brokerType,
                    environment: env,
                    credentials,
                }),
            });

            const payload = await res.json().catch(() => ({})) as { id?: string; detail?: string };
            if (!res.ok) throw new Error(payload.detail || "Failed to add connection.");

            onSuccess();
            onClose();
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : "An error occurred.");
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="relative w-full max-w-md glass-card rounded-2xl p-6 shadow-2xl animate-in fade-in zoom-in duration-300">
                <h2 className="mb-1 text-lg font-semibold text-foreground">Add Broker Connection</h2>
                <p className="mb-5 text-xs text-muted-foreground">
                    Credentials are encrypted at rest. <span className="font-medium text-foreground">Never shared.</span>
                </p>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <SegmentedControl
                        options={[
                            { label: "Alpaca", value: "alpaca" },
                            { label: "IBKR", value: "ibkr" }
                        ]}
                        value={brokerType}
                        onChange={(v) => setBrokerType(v as BrokerType)}
                    />

                    <div className="space-y-1.5">
                        <label className="block text-xs font-semibold text-muted-foreground uppercase tracking-wider">Nickname</label>
                        <Input
                            required
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="e.g. My IRA, Algo Paper"
                            className="bg-background/50"
                        />
                    </div>

                    {brokerType === "alpaca" && (
                        <div className="space-y-4 animate-in slide-in-from-top-2 duration-300">
                            <SegmentedControl
                                options={[
                                    { label: "📄 Paper", value: "paper" },
                                    { label: "⚡ Live", value: "live" }
                                ]}
                                value={env}
                                onChange={(v) => setEnv(v as Environment)}
                            />
                            <div className="space-y-1.5">
                                <label className="block text-xs font-semibold text-muted-foreground uppercase tracking-wider">API Key</label>
                                <Input
                                    required
                                    value={apiKey}
                                    onChange={(e) => setApiKey(e.target.value)}
                                    placeholder="PKXXXXXXXXXXXXXXXXXX"
                                    className="font-mono bg-background/50"
                                />
                            </div>
                            <div className="space-y-1.5">
                                <label className="block text-xs font-semibold text-muted-foreground uppercase tracking-wider">API Secret</label>
                                <Input
                                    required
                                    type="password"
                                    value={apiSecret}
                                    onChange={(e) => setApiSecret(e.target.value)}
                                    placeholder="••••••••••••••••••••"
                                    className="font-mono bg-background/50"
                                />
                            </div>
                        </div>
                    )}

                    {brokerType === "ibkr" && (
                        <div className="space-y-4 animate-in slide-in-from-top-2 duration-300">
                            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-500/90 backdrop-blur-sm">
                                <span className="font-bold">Requires IB Gateway</span> running and accessible. Use Tailscale for remote access.
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="col-span-2 space-y-1.5">
                                    <label className="block text-xs font-semibold text-muted-foreground uppercase tracking-wider">Host / IP</label>
                                    <Input
                                        required
                                        value={ibkrHost}
                                        onChange={(e) => setIbkrHost(e.target.value)}
                                        placeholder="127.0.0.1 or Tailscale IP"
                                        className="font-mono bg-background/50"
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="block text-xs font-semibold text-muted-foreground uppercase tracking-wider">Port</label>
                                    <Input
                                        required
                                        value={ibkrPort}
                                        onChange={(e) => setIbkrPort(e.target.value)}
                                        placeholder="7497"
                                        className="font-mono bg-background/50"
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="block text-xs font-semibold text-muted-foreground uppercase tracking-wider">Client ID</label>
                                    <Input
                                        required
                                        value={ibkrClientId}
                                        onChange={(e) => setIbkrClientId(e.target.value)}
                                        placeholder="10"
                                        className="font-mono bg-background/50"
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {error && (
                        <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive animate-in shake duration-300">
                            <AlertTriangle className="h-4 w-4 shrink-0" />
                            {error}
                        </div>
                    )}

                    <div className="flex gap-2 pt-2">
                        <Button
                            type="button"
                            variant="outline"
                            onClick={onClose}
                            className="flex-1"
                        >
                            Cancel
                        </Button>
                        <Button
                            type="submit"
                            disabled={submitting}
                            className="flex-1"
                        >
                            {submitting ? "Connecting…" : "Add Connection"}
                        </Button>
                    </div>
                </form>
            </div>
        </div>
    );
}

// ─── Main Component ──────────────────────────────────────────────────────────
interface BrokerConnectionsProps {
    accessToken: string | null | undefined;
}

export default function BrokerConnections({ accessToken }: BrokerConnectionsProps) {
    const [connections, setConnections] = useState<BrokerConnection[]>([]);
    const [balance, setBalance] = useState<PortfolioBalance | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [showAddModal, setShowAddModal] = useState(false);
    const [removingId, setRemovingId] = useState<string | null>(null);
    const [togglingId, setTogglingId] = useState<string | null>(null);

    const base = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
    const authHeaders = {
        "Content-Type": "application/json",
        ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    };

    const fetchAll = useCallback(async () => {
        setLoading(true);
        setError("");
        try {
            const [connRes, balRes] = await Promise.all([
                fetch(`${base}/brokers`, { headers: authHeaders }),
                fetch(`${base}/portfolio/balance`, { headers: authHeaders }),
            ]);

            if (!connRes.ok) throw new Error("Failed to load broker connections.");
            const connData = await connRes.json() as BrokerConnection[];
            setConnections(Array.isArray(connData) ? connData : []);

            if (balRes.ok) {
                const balData = await balRes.json() as PortfolioBalance;
                setBalance(balData);
            }
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : "Could not load broker data.");
        } finally {
            setLoading(false);
        }
    }, [accessToken, base]);

    useEffect(() => { void fetchAll(); }, [fetchAll]);

    const handleToggle = async (conn: BrokerConnection) => {
        setTogglingId(conn.id);
        try {
            await fetch(`${base}/brokers/${conn.id}/toggle`, {
                method: "PATCH",
                headers: authHeaders,
            });
            await fetchAll();
        } finally {
            setTogglingId(null);
        }
    };

    const handleRemove = async (id: string) => {
        if (!confirm("Remove this broker connection? This cannot be undone.")) return;
        setRemovingId(id);
        try {
            await fetch(`${base}/brokers/${id}`, { method: "DELETE", headers: authHeaders });
            await fetchAll();
        } finally {
            setRemovingId(null);
        }
    };

    const formatCurrency = (v: number) =>
        new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(v);

    return (
        <section className="space-y-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-bold text-foreground tracking-tight">Broker Connections</h2>
                    <p className="mt-1 text-sm text-muted-foreground">
                        Broker credentials are <span className="text-foreground font-medium">AES-256 encrypted</span>. Risk limits apply globally.
                    </p>
                </div>
                <div className="flex gap-3 w-full sm:w-auto">
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => void fetchAll()}
                        className="flex-1 sm:flex-none h-10"
                    >
                        <RefreshCw className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
                        Refresh
                    </Button>
                    <Button
                        size="sm"
                        onClick={() => setShowAddModal(true)}
                        className="flex-1 sm:flex-none h-10"
                    >
                        <Plus className="h-4 w-4 mr-2" />
                        Add Broker
                    </Button>
                </div>
            </div>

            {balance && (
                <article className="glass-card rounded-2xl p-6 border-primary/20 bg-primary/5">
                    <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
                        <div className="space-y-1">
                            <p className="text-xs font-bold uppercase tracking-[0.2em] text-muted-foreground/80">Net Combined Equity</p>
                            <p className="text-4xl font-extrabold text-foreground tracking-tight">{formatCurrency(balance.total_equity)}</p>
                            {balance.last_updated && (
                                <p className="text-[11px] text-muted-foreground flex items-center gap-1.5">
                                    <Timer className="h-3 w-3" />
                                    Last synchronized {new Date(balance.last_updated).toLocaleTimeString()}
                                </p>
                            )}
                        </div>
                        <div className="flex flex-wrap gap-4 md:gap-8 bg-background/30 backdrop-blur-md rounded-xl p-4 md:p-5 border border-border/50">
                            {balance.breakdown.map((bk) => (
                                <div key={bk.source} className="flex flex-col">
                                    <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{bk.source}</span>
                                    <span className="text-sm font-semibold text-foreground mt-0.5">{formatCurrency(bk.equity)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </article>
            )}

            {error && (
                <div className="flex items-center gap-3 rounded-xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive animate-in fade-in duration-300">
                    <AlertTriangle className="h-5 w-5 shrink-0" />
                    <div>
                        <p className="font-semibold">Connectivity Error</p>
                        <p className="opacity-90">{error}</p>
                    </div>
                </div>
            )}

            {!loading && !error && connections.length === 0 && (
                <article className="glass-card flex flex-col items-center gap-5 rounded-2xl py-16 text-center border-dashed">
                    <div className="relative">
                        <Server className="h-12 w-12 text-muted-foreground/30" />
                        <div className="absolute -top-1 -right-1 h-4 w-4 bg-primary rounded-full animate-pulse" />
                    </div>
                    <div className="max-w-xs">
                        <p className="text-lg font-bold text-foreground">Secure Connections</p>
                        <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                            No active broker links. Connect Alpaca or IBKR to enable cross-account execution.
                        </p>
                    </div>
                    <Button
                        onClick={() => setShowAddModal(true)}
                        size="lg"
                        className="rounded-full px-8"
                    >
                        <Plus className="h-4 w-4 mr-2" />
                        Get Started
                    </Button>
                </article>
            )}

            {!loading && connections.length > 0 && (
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                    {connections.map((conn) => (
                        <article
                            key={conn.id}
                            className={cn(
                                "glass-card rounded-2xl p-5 transition-all duration-300",
                                !conn.is_active && "opacity-60 grayscale-[0.5] border-dashed"
                            )}
                        >
                            <div className="flex items-start justify-between gap-3 mb-4">
                                <div className="space-y-1.5">
                                    <div className="flex items-center gap-2">
                                        <StatusPip active={conn.is_active} />
                                        <span className="font-bold text-foreground tracking-tight truncate max-w-[120px]">{conn.name}</span>
                                    </div>
                                    <p className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground">
                                        Joined {new Date(conn.created_at).toLocaleDateString()}
                                    </p>
                                </div>
                                <div className="flex flex-col items-end gap-1.5">
                                    <BrokerBadge type={conn.broker_type} />
                                    <EnvBadge env={conn.environment} />
                                </div>
                            </div>

                            <div className="flex items-center gap-2 pt-2 mt-4 border-t border-border/40">
                                <Button
                                    variant={conn.is_active ? "outline" : "default"}
                                    size="xs"
                                    onClick={() => void handleToggle(conn)}
                                    disabled={togglingId === conn.id}
                                    className={cn(
                                        "flex-1 h-8 rounded-lg",
                                        !conn.is_active && "bg-emerald-600 hover:bg-emerald-700 text-white"
                                    )}
                                >
                                    {conn.is_active ? (
                                        <><WifiOff className="h-3.5 w-3.5 mr-1.5" /> Disable</>
                                    ) : (
                                        <><Zap className="h-3.5 w-3.5 mr-1.5" /> Enable</>
                                    )}
                                </Button>
                                <Button
                                    variant="ghost"
                                    size="icon-xs"
                                    onClick={() => void handleRemove(conn.id)}
                                    disabled={removingId === conn.id}
                                    className="h-8 w-8 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                                >
                                    <Trash2 className="h-3.5 w-3.5" />
                                </Button>
                            </div>
                        </article>
                    ))}
                </div>
            )}

            {!loading && !error && connections.length > 0 && (
                <div className="glass-card p-4 rounded-xl border-emerald-500/20 bg-emerald-500/5 animate-in slide-in-from-bottom-2 duration-500">
                    <p className="flex items-center gap-2.5 text-xs font-medium text-emerald-600 dark:text-emerald-400">
                        <CheckCircle2 className="h-4 w-4" />
                        Multi-broker aggregation active with {connections.filter((c) => c.is_active).length} active link(s).
                    </p>
                </div>
            )}

            {showAddModal && (
                <AddConnectionModal
                    onClose={() => setShowAddModal(false)}
                    onSuccess={() => void fetchAll()}
                    accessToken={accessToken}
                />
            )}
        </section>
    );
}
