"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Plus, RefreshCw, Server, Trash2, WifiOff, Zap } from "lucide-react";

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
function StatusPip({ active }: { active: boolean }) {
    return (
        <span
            className={`inline-block h-2 w-2 rounded-full ${active ? "bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.7)]" : "bg-zinc-500"
                }`}
        />
    );
}

function BrokerBadge({ type }: { type: BrokerType }) {
    const styles: Record<BrokerType, string> = {
        alpaca: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-300",
        ibkr: "bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300",
    };
    const labels: Record<BrokerType, string> = {
        alpaca: "Alpaca",
        ibkr: "IBKR",
    };
    return (
        <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${styles[type]}`}>
            {labels[type]}
        </span>
    );
}

function EnvBadge({ env }: { env: Environment }) {
    return env === "live" ? (
        <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
            Live
        </span>
    ) : (
        <span className="rounded-full bg-zinc-200 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
            Paper
        </span>
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
    // Alpaca fields
    const [apiKey, setApiKey] = useState("");
    const [apiSecret, setApiSecret] = useState("");
    // IBKR fields
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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="relative w-full max-w-md rounded-2xl border border-border bg-card p-6 shadow-2xl">
                <h2 className="mb-1 text-lg font-semibold text-foreground">Add Broker Connection</h2>
                <p className="mb-5 text-xs text-muted-foreground">
                    Credentials are encrypted at rest. <span className="font-medium text-foreground">Never shared.</span>
                </p>

                <form onSubmit={handleSubmit} className="space-y-4">
                    {/* Broker type toggle */}
                    <div className="flex gap-2 rounded-xl border border-border p-1">
                        {(["alpaca", "ibkr"] as BrokerType[]).map((t) => (
                            <button
                                key={t}
                                type="button"
                                onClick={() => setBrokerType(t)}
                                className={`flex-1 rounded-lg py-1.5 text-sm font-semibold transition ${brokerType === t
                                        ? "bg-primary text-primary-foreground"
                                        : "text-muted-foreground hover:text-foreground"
                                    }`}
                            >
                                {t === "alpaca" ? "Alpaca" : "IBKR"}
                            </button>
                        ))}
                    </div>

                    {/* Name */}
                    <div>
                        <label className="mb-1 block text-xs font-medium text-muted-foreground">Nickname</label>
                        <input
                            required
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="e.g. My IRA, Algo Paper"
                            className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-primary"
                        />
                    </div>

                    {/* Alpaca-specific */}
                    {brokerType === "alpaca" && (
                        <>
                            <div className="flex gap-2 rounded-xl border border-border p-1">
                                {(["paper", "live"] as Environment[]).map((e) => (
                                    <button
                                        key={e}
                                        type="button"
                                        onClick={() => setEnv(e)}
                                        className={`flex-1 rounded-lg py-1.5 text-sm font-semibold transition ${env === e ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                                            }`}
                                    >
                                        {e === "paper" ? "📄 Paper" : "⚡ Live"}
                                    </button>
                                ))}
                            </div>
                            <div>
                                <label className="mb-1 block text-xs font-medium text-muted-foreground">API Key</label>
                                <input
                                    required
                                    value={apiKey}
                                    onChange={(e) => setApiKey(e.target.value)}
                                    placeholder="PKXXXXXXXXXXXXXXXXXX"
                                    className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground font-mono outline-none focus:border-primary"
                                />
                            </div>
                            <div>
                                <label className="mb-1 block text-xs font-medium text-muted-foreground">API Secret</label>
                                <input
                                    required
                                    type="password"
                                    value={apiSecret}
                                    onChange={(e) => setApiSecret(e.target.value)}
                                    placeholder="••••••••••••••••••••"
                                    className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground font-mono outline-none focus:border-primary"
                                />
                            </div>
                        </>
                    )}

                    {/* IBKR-specific */}
                    {brokerType === "ibkr" && (
                        <>
                            <div className="rounded-lg border border-amber-300/50 bg-amber-50/50 px-3 py-2 text-xs text-amber-700 dark:border-amber-700/30 dark:bg-amber-900/20 dark:text-amber-300">
                                <strong>Requires IB Gateway</strong> running and accessible. Use Tailscale for remote access.
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="col-span-2">
                                    <label className="mb-1 block text-xs font-medium text-muted-foreground">Host / IP</label>
                                    <input
                                        required
                                        value={ibkrHost}
                                        onChange={(e) => setIbkrHost(e.target.value)}
                                        placeholder="127.0.0.1 or Tailscale IP"
                                        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground font-mono outline-none focus:border-primary"
                                    />
                                </div>
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-muted-foreground">Port</label>
                                    <input
                                        required
                                        value={ibkrPort}
                                        onChange={(e) => setIbkrPort(e.target.value)}
                                        placeholder="7497"
                                        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground font-mono outline-none focus:border-primary"
                                    />
                                </div>
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-muted-foreground">Client ID</label>
                                    <input
                                        required
                                        value={ibkrClientId}
                                        onChange={(e) => setIbkrClientId(e.target.value)}
                                        placeholder="10"
                                        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground font-mono outline-none focus:border-primary"
                                    />
                                </div>
                            </div>
                        </>
                    )}

                    {error && (
                        <p className="flex items-center gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                            <AlertTriangle className="h-4 w-4 shrink-0" />
                            {error}
                        </p>
                    )}

                    <div className="flex gap-2 pt-1">
                        <button
                            type="button"
                            onClick={onClose}
                            className="flex-1 rounded-lg border border-border py-2 text-sm font-semibold text-foreground transition hover:bg-muted/50"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={submitting}
                            className="flex-1 rounded-lg bg-primary py-2 text-sm font-semibold text-primary-foreground transition hover:opacity-90 disabled:opacity-60"
                        >
                            {submitting ? "Connecting…" : "Add Connection"}
                        </button>
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
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [accessToken]);

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
        <section className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold text-foreground">Broker Connections</h2>
                    <p className="mt-0.5 text-xs text-muted-foreground">
                        All credentials are encrypted at rest. Risk limits apply across all accounts.
                    </p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => void fetchAll()}
                        className="flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-foreground transition hover:bg-muted/50"
                    >
                        <RefreshCw className="h-3.5 w-3.5" />
                        Refresh
                    </button>
                    <button
                        onClick={() => setShowAddModal(true)}
                        className="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground transition hover:opacity-90"
                    >
                        <Plus className="h-3.5 w-3.5" />
                        Add Broker
                    </button>
                </div>
            </div>

            {/* Aggregated Balance */}
            {balance && (
                <article className="rounded-xl border border-primary/30 bg-primary/5 p-4">
                    <div className="flex items-center justify-between gap-4">
                        <div>
                            <p className="text-xs uppercase tracking-wide text-muted-foreground">Total Equity (All Accounts)</p>
                            <p className="mt-0.5 text-3xl font-bold text-foreground">{formatCurrency(balance.total_equity)}</p>
                            {balance.last_updated && (
                                <p className="mt-1 text-[11px] text-muted-foreground">
                                    Updated {new Date(balance.last_updated).toLocaleTimeString()}
                                </p>
                            )}
                        </div>
                        <div className="hidden sm:block space-y-1 text-right">
                            {balance.breakdown.map((bk) => (
                                <div key={bk.source} className="text-xs text-muted-foreground">
                                    <span className="font-medium text-foreground">{bk.source}</span>
                                    {" — "}
                                    {formatCurrency(bk.equity)}
                                </div>
                            ))}
                        </div>
                    </div>
                </article>
            )}

            {/* Error */}
            {error && (
                <p className="flex items-center gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                    <AlertTriangle className="h-4 w-4 shrink-0" />
                    {error}
                </p>
            )}

            {/* Loading */}
            {loading && !error && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Loading connections…
                </div>
            )}

            {/* Empty state */}
            {!loading && !error && connections.length === 0 && (
                <article className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-border py-10 text-center">
                    <Server className="h-8 w-8 text-muted-foreground/50" />
                    <div>
                        <p className="font-medium text-foreground">No broker connections yet</p>
                        <p className="mt-1 text-xs text-muted-foreground">
                            Add Alpaca or IBKR to aggregate equity and enable multi-broker risk checks.
                        </p>
                    </div>
                    <button
                        onClick={() => setShowAddModal(true)}
                        className="flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition hover:opacity-90"
                    >
                        <Plus className="h-4 w-4" />
                        Add First Broker
                    </button>
                </article>
            )}

            {/* Connection cards */}
            {!loading && connections.length > 0 && (
                <div className="grid gap-3 sm:grid-cols-2">
                    {connections.map((conn) => (
                        <article
                            key={conn.id}
                            className={`rounded-xl border p-4 transition ${conn.is_active
                                    ? "border-border/70 bg-card/60"
                                    : "border-dashed border-border/50 bg-muted/20 opacity-70"
                                }`}
                        >
                            <div className="flex items-start justify-between gap-2">
                                <div className="flex items-center gap-2">
                                    <StatusPip active={conn.is_active} />
                                    <span className="font-semibold text-foreground truncate max-w-[140px]">{conn.name}</span>
                                </div>
                                <div className="flex shrink-0 items-center gap-1.5">
                                    <BrokerBadge type={conn.broker_type} />
                                    <EnvBadge env={conn.environment} />
                                </div>
                            </div>

                            <p className="mt-2 text-[11px] text-muted-foreground">
                                Added {new Date(conn.created_at).toLocaleDateString()}
                            </p>

                            {/* Actions */}
                            <div className="mt-3 flex gap-2">
                                <button
                                    onClick={() => void handleToggle(conn)}
                                    disabled={togglingId === conn.id}
                                    className={`flex items-center gap-1 rounded-md px-2.5 py-1 text-xs font-semibold transition ${conn.is_active
                                            ? "border border-border text-muted-foreground hover:text-foreground hover:bg-muted/50"
                                            : "bg-emerald-600 text-white hover:bg-emerald-700"
                                        }`}
                                >
                                    {conn.is_active ? (
                                        <><WifiOff className="h-3 w-3" /> Disable</>
                                    ) : (
                                        <><Zap className="h-3 w-3" /> Enable</>
                                    )}
                                </button>
                                <button
                                    onClick={() => void handleRemove(conn.id)}
                                    disabled={removingId === conn.id}
                                    className="ml-auto flex items-center gap-1 rounded-md border border-destructive/40 px-2.5 py-1 text-xs font-semibold text-destructive transition hover:bg-destructive/10"
                                >
                                    <Trash2 className="h-3 w-3" />
                                    {removingId === conn.id ? "Removing…" : "Remove"}
                                </button>
                            </div>
                        </article>
                    ))}
                </div>
            )}

            {/* Success indicator inline */}
            {!loading && !error && connections.length > 0 && (
                <p className="flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    Aggregate risk is active across {connections.filter((c) => c.is_active).length} connected account(s).
                </p>
            )}

            {/* Add modal */}
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
