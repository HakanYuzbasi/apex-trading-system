"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  AlertTriangle,
  BarChart3,
  DollarSign,
  Gauge,
  LogOut,
  Moon,
  RefreshCw,
  Server,
  ShieldCheck,
  Sun,
  Timer,
  TrendingDown,
  TrendingUp,
  Wifi,
  WifiOff,
} from "lucide-react";
import {
  useCockpitData,
  useMetrics,
  type CockpitAlert,
  type CockpitDerivative,
  type CockpitPosition,
  type ReplayInspectionResponse,
  type SleeveAttribution,
} from "@/lib/api";
import { useTheme } from "@/components/theme/ThemeProvider";
import { useAuthContext } from "@/components/auth/AuthProvider";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import AlertsFeed from "@/components/dashboard/AlertsFeed";
import ControlsPanel from "@/components/dashboard/ControlsPanel";
import EquityPanel from "@/components/dashboard/EquityPanel";
import PositionsTable from "@/components/dashboard/PositionsTable";
import ReplayInspectorPanel from "@/components/dashboard/ReplayInspectorPanel";
import ExplainableAIChart from "@/components/dashboard/ExplainableAIChart";
import MandateCopilotPanel from "@/components/dashboard/MandateCopilotPanel";
import SocialGovernorPanel, { type SocialGovernorPanelProps } from "@/components/dashboard/SocialGovernorPanel";
import WalkForwardPanel from "@/components/dashboard/WalkForwardPanel";
import PortfolioHeatPanel from "@/components/dashboard/PortfolioHeatPanel";
import RegimeSharpePanel from "@/components/dashboard/RegimeSharpePanel";
import FactorPnlPanel from "@/components/dashboard/FactorPnlPanel";
import CorrelationHeatmapPanel from "@/components/dashboard/CorrelationHeatmapPanel";
import RegimeBacktestPanel from "@/components/dashboard/RegimeBacktestPanel";
import WssMetricsWidget from "@/components/dashboard/WssMetricsWidget";
import ICReportPanel from "@/components/dashboard/ICReportPanel";
import MissionControlPanel from "@/components/dashboard/MissionControlPanel";
import TradePostMortemPanel from "@/components/dashboard/TradePostMortemPanel";
import EquityCurvePanel from "@/components/dashboard/EquityCurvePanel";
import StressPanel from "@/components/dashboard/StressPanel";
import AttributionPanel from "@/components/dashboard/AttributionPanel";
import MissedOpportunitiesPanel from "@/components/dashboard/MissedOpportunitiesPanel";
import AdvancedMetricsPanel from "@/components/dashboard/AdvancedMetricsPanel";
import TcaReportPanel from "@/components/dashboard/TcaReportPanel";
import StressScenariosPanel from "@/components/dashboard/StressScenariosPanel";
import OrderRejectionsPanel from "@/components/dashboard/OrderRejectionsPanel";
import FeatureIcPanel from "@/components/dashboard/FeatureIcPanel";
import AbGatePanel from "@/components/dashboard/AbGatePanel";
import SentimentHealthPanel, { type VetoDetail } from "@/components/dashboard/SentimentHealthPanel";
import PortfolioWeightsPanel from "@/components/dashboard/PortfolioWeightsPanel";
import HmmRegimePanel from "@/components/dashboard/HmmRegimePanel";
import AlertHistoryPanel from "@/components/dashboard/AlertHistoryPanel";
import PaperAccountPanel from "@/components/dashboard/PaperAccountPanel";
import ModelRegistryPanel from "@/components/dashboard/ModelRegistryPanel";
import CrossAssetPairsPanel from "@/components/dashboard/CrossAssetPairsPanel";
import IVCrushPanel from "@/components/dashboard/IVCrushPanel";
import StrategyAllocationPanel, { type StrategyAllocationData } from "@/components/dashboard/StrategyAllocationPanel";
import SocialPulsePanel, { type SocialPulseData } from "@/components/dashboard/SocialPulsePanel";
import BriefingArchivePanel from "@/components/dashboard/BriefingArchivePanel";
import BrokerModePanel from "@/components/dashboard/BrokerModePanel";
import BrokerReconciliationPanel, { type BrokerPosition } from "@/components/dashboard/BrokerReconciliationPanel";
import SignalHeatmap from "@/components/SignalHeatmap";
import { Skeleton } from "@/components/ui/skeleton";
import { useWebSocket } from "@/hooks/useWebSocket";
import {
  sanitizeCount,
  sanitizeExecutionMetrics,
  sanitizeMoney,
} from "@/lib/metricGuards";
import {
  MAX_POSITIONS,
  DRAWDOWN_BUDGET_PCT,
  SHARPE_TARGET,
  WIN_RATE_TARGET,
  RETURN_CYCLE_TARGET,
  TRADE_CYCLE_TARGET,
  EDGE_CAPTURE_TARGET,
} from "@/lib/constants";
import {
  clampPct,
  formatCurrency,
  formatCurrencyWithCents,
  formatCompactCurrency,
  formatPct,
  normalizeDrawdownPct,
  formatSleeveLabel,
  sortIndicator,
} from "@/lib/formatters";

type DashboardTab = "trading" | "mandate" | "social" | "walkforward" | "heat" | "diagnostics" | "factorpnl" | "corrmap" | "regimebt" | "wssmetrics" | "icreport" | "missioncontrol" | "postmortem" | "equitycurve" | "stress" | "attribution" | "missed" | "advancedmetrics" | "tca" | "stressscenarios" | "orderrejections" | "featureic" | "abgate" | "portfolioweights" | "hmmregime" | "alerts" | "paperaccount" | "modelregistry" | "crossassetpairs" | "ivcrush" | "briefings" | "brokersync";
type LensKey = "performance" | "risk" | "execution";
type SortDirection = "asc" | "desc";
type PositionSortKey = "symbol" | "qty" | "entry" | "current" | "pnl" | "pnl_pct" | "signal_direction";
type SeverityFilter = "all" | "critical" | "warning" | "info";

type LensRow = {
  label: string;
  value: string;
  hint: string;
  tone?: "neutral" | "positive" | "negative";
};

type LensBar = {
  label: string;
  value: number;
  targetLabel: string;
  tone?: "neutral" | "positive" | "negative";
};

type LensModel = {
  title: string;
  subtitle: string;
  rows: LensRow[];
  bars: LensBar[];
};

type ReplayTarget = {
  kind: "symbol" | "plan";
  label: string;
  requestUrl: string;
  symbol?: string | null;
  planId?: string | null;
};



function toneClass(tone: LensRow["tone"]): string {
  if (tone === "positive") return "text-positive";
  if (tone === "negative") return "text-negative";
  return "text-foreground";
}

function barToneClass(tone: LensBar["tone"]): string {
  if (tone === "positive") return "bg-positive";
  if (tone === "negative") return "bg-negative";
  return "bg-primary";
}

function severityBadgeClass(severity: CockpitAlert["severity"]): string {
  if (severity === "critical") return "bg-negative/15 text-negative";
  if (severity === "warning") return "bg-warning/15 text-warning";
  return "bg-primary/15 text-primary";
}

function comparePositions(a: CockpitPosition, b: CockpitPosition, key: PositionSortKey): number {
  if (key === "symbol" || key === "signal_direction") {
    return String(a[key]).localeCompare(String(b[key]));
  }
  return Number(a[key]) - Number(b[key]);
}

function positionRowKey(position: CockpitPosition): string {
  const symbol = String(position.symbol ?? "").trim().toUpperCase();
  const source = String(position.source_id ?? "").trim()
    || String(position.broker_type ?? "").trim().toLowerCase()
    || "state";
  const sideRaw = String(position.side ?? (position.qty < 0 ? "SHORT" : "LONG")).trim().toUpperCase();
  const side = sideRaw === "SHORT" ? "SHORT" : "LONG";
  // Preserve fractional qty for crypto in the hash key (Math.trunc(0.1234) = 0 → wrong dedup key)
  const rawQtyNum = Number.isFinite(Number(position.qty)) ? Number(position.qty) : 0;
  const qtyForHash = String(symbol).includes("/") || String(symbol).toUpperCase().startsWith("CRYPTO:")
    ? rawQtyNum.toFixed(6)
    : String(Math.trunc(rawQtyNum));
  const securityType = String((position as unknown as Record<string, unknown>).security_type ?? "").trim().toUpperCase();
  const expiry = String((position as unknown as Record<string, unknown>).expiry ?? "").trim().toUpperCase();
  const strike = Number.isFinite(Number((position as unknown as Record<string, unknown>).strike))
    ? Number((position as unknown as Record<string, unknown>).strike).toFixed(4)
    : "0.0000";
  const right = String((position as unknown as Record<string, unknown>).right ?? "").trim().toUpperCase();
  return `${symbol}|${source}|${side}|${securityType}|${expiry}|${strike}|${right}|${qtyForHash}`;
}


function readinessClass(state: "ok" | "warn" | "down"): string {
  if (state === "ok") return "bg-positive/15 text-positive";
  if (state === "warn") return "bg-warning/15 text-warning";
  return "bg-negative/15 text-negative";
}


export default function Dashboard({ isPublic = false }: { isPublic?: boolean }) {
  const router = useRouter();
  const { logout } = useAuthContext();
  const { isConnected: wsConnected, lastMessage: wsMessage, reconnectAttempt } = useWebSocket(isPublic);
  const { metrics, isLoading: metricsLoading, isError: metricsError, error: metricsFetchError } = useMetrics(isPublic);
  const { data: cockpit, isLoading: cockpitLoading, isError: cockpitError, error: cockpitFetchError } = useCockpitData(isPublic);
  const { theme, toggleTheme } = useTheme();

  // Alert management state
  const [dismissedAlerts, setDismissedAlerts] = useState<Record<string, boolean>>({});
  const [alertFilter, setAlertFilter] = useState<SeverityFilter>("all");

  // UI state
  const [themeMounted, setThemeMounted] = useState(false);
  const [activeLens, setActiveLens] = useState<LensKey>("performance");
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Multi-broker aggregated equity
  const [aggregatedEquity, setAggregatedEquity] = useState<number | null>(null);
  const [brokerCount, setBrokerCount] = useState<number>(0);

  // Account selector — broker source list and selected account
  const [brokerSources, setBrokerSources] = useState<{ id: string; name: string; broker_type: string; environment: string }[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);

  // Position table sorting state
  const [sortKey, setSortKey] = useState<PositionSortKey>("symbol");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");
  const [positionsSnapshot, setPositionsSnapshot] = useState<CockpitPosition[]>([]);
  const [derivativesSnapshot, setDerivativesSnapshot] = useState<CockpitDerivative[]>([]);
  const [sleevesSnapshot, setSleevesSnapshot] = useState<SleeveAttribution[]>([]);
  const [replayTarget, setReplayTarget] = useState<ReplayTarget | null>(null);
  const [replayData, setReplayData] = useState<ReplayInspectionResponse | null>(null);
  const [replayLoading, setReplayLoading] = useState(false);
  const [replayError, setReplayError] = useState("");

  // Tab navigation
  const [activeTab, setActiveTab] = useState<DashboardTab>("trading");

  const [sessionExpiredRedirecting, setSessionExpiredRedirecting] = useState(false);

  // ... existing state ...

  const handleSessionExpired = useCallback(() => {
    if (isPublic || sessionExpiredRedirecting) {
      return;
    }
    setSessionExpiredRedirecting(true);
    logout();
    router.replace("/login?reason=session_expired");
  }, [isPublic, logout, router, sessionExpiredRedirecting]);

  // Parse WebSocket message
  const wsData = wsMessage?.type === "state_update" ? wsMessage : null;

  useEffect(() => {
    if (!replayTarget?.requestUrl) {
      setReplayData(null);
      setReplayLoading(false);
      setReplayError("");
      return;
    }

    let cancelled = false;
    const controller = new AbortController();

    setReplayLoading(true);
    setReplayError("");

    fetch(replayTarget.requestUrl, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (response) => {
        const payload = await response.json().catch(() => null);
        if (!response.ok) {
          const detail = payload && typeof payload === "object" && "detail" in payload
            ? String((payload as Record<string, unknown>).detail)
            : `Replay inspector failed (${response.status})`;
          throw new Error(detail);
        }
        return payload as ReplayInspectionResponse;
      })
      .then((payload) => {
        if (!cancelled) {
          setReplayData(payload);
        }
      })
      .catch((error: unknown) => {
        if (cancelled || (error instanceof DOMException && error.name === "AbortError")) {
          return;
        }
        setReplayData(null);
        setReplayError(error instanceof Error ? error.message : "Failed to load replay chain.");
      })
      .finally(() => {
        if (!cancelled) {
          setReplayLoading(false);
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [replayTarget]);

  // Extract SHAP values from WS or fall back to mock data for demonstration
  const activeShapData = ((wsData as unknown as Record<string, unknown>)?.shap_values as Record<string, number>) ?? {
    "Price Momentum (Feature #2)": 0.45,
    "Volatility Profile (Feature #1)": -0.12,
    "RSI State (Feature #4)": 0.08,
    "MACD Action (Feature #0)": -0.05,
    "Statistical Z-Score (Feature #3)": 0.02
  };

  // Merge Metrics: Prefer WS data, fallback to REST
  const mergedMetrics = useMemo(() => {
    const cockpitStatus = cockpit?.status;
    const baseline = cockpitStatus
      ? {
        status: cockpitStatus.state_fresh,
        timestamp: cockpitStatus.timestamp,
        capital: cockpitStatus.capital,
        starting_capital: cockpitStatus.starting_capital ?? 0,
        daily_pnl: cockpitStatus.daily_pnl,
        total_pnl: cockpitStatus.total_pnl,
        max_drawdown: cockpitStatus.max_drawdown,
        sharpe_ratio: cockpitStatus.sharpe_ratio,
        sortino_ratio: cockpitStatus.sortino_ratio ?? 0,
        calmar_ratio: cockpitStatus.calmar_ratio ?? 0,
        profit_factor: cockpitStatus.profit_factor ?? 0,
        alpha_retention: cockpitStatus.alpha_retention ?? 0,
        win_rate: cockpitStatus.win_rate,
        open_positions: cockpitStatus.open_positions,
        option_positions: cockpitStatus.option_positions,
        open_positions_total: cockpitStatus.open_positions_total,
        trades_count: cockpitStatus.total_trades,
      }
      : metrics;
    if (!wsData) {
      return baseline;
    }

    const baselineRecord = (baseline ?? {}) as Record<string, unknown>;
    const wsCapital = sanitizeMoney(
      wsData.aggregated_equity ?? wsData.total_equity ?? wsData.capital,
      Number.NaN,
    );
    const safeWsValue = (primary: unknown, fallback: unknown): number => {
      if (primary !== undefined && primary !== null) {
        const num = Number(primary);
        return Number.isFinite(num) ? num : 0;
      }
      const fNum = Number(fallback);
      return Number.isFinite(fNum) ? fNum : 0;
    };

    return {
      status: true,
      timestamp: wsData.timestamp ?? baselineRecord.timestamp ?? null,
      capital: Number.isFinite(wsCapital) ? wsCapital : sanitizeMoney(baselineRecord.capital, 0),
      starting_capital: sanitizeMoney(
        wsData.starting_capital ?? wsData.initial_capital,
        sanitizeMoney(baselineRecord.starting_capital, 0),
      ),
      daily_pnl: safeWsValue(wsData.daily_pnl, baselineRecord.daily_pnl),
      total_pnl: safeWsValue(wsData.total_pnl, baselineRecord.total_pnl),
      max_drawdown: safeWsValue(wsData.max_drawdown, baselineRecord.max_drawdown),
      sharpe_ratio: safeWsValue(wsData.sharpe_ratio, baselineRecord.sharpe_ratio),
      sortino_ratio: safeWsValue(wsData.sortino_ratio, baselineRecord.sortino_ratio),
      calmar_ratio: safeWsValue(wsData.calmar_ratio, baselineRecord.calmar_ratio),
      profit_factor: safeWsValue(wsData.profit_factor, baselineRecord.profit_factor),
      alpha_retention: safeWsValue(wsData.alpha_retention, baselineRecord.alpha_retention),
      win_rate: safeWsValue(wsData.win_rate, baselineRecord.win_rate),
      open_positions: safeWsValue(wsData.open_positions, baselineRecord.open_positions),
      option_positions: safeWsValue(wsData.option_positions, baselineRecord.option_positions),
      open_positions_total: safeWsValue(wsData.open_positions_total ?? wsData.open_positions, baselineRecord.open_positions_total),
      trades_count: safeWsValue(wsData.total_trades, baselineRecord.trades_count),
    };
  }, [wsData, metrics, cockpit?.status]);

  // Fetch aggregated equity from multi-broker balance endpoint
  useEffect(() => {
    const fetchAggregated = async () => {
      try {
        const res = await fetch("/api/v1/portfolio/balance", { cache: "no-store" });
        if (res.status === 401) {
          handleSessionExpired();
          return;
        }
        if (res.ok) {
          const data = await res.json() as { total_equity: number; breakdown?: { source: string }[] };
          setAggregatedEquity(sanitizeMoney(data.total_equity, 0));
          setBrokerCount(Math.max(1, sanitizeCount(data.breakdown?.length ?? 1, 1)));
        }
      } catch {
        // silently skip — not critical
      }
    };
    void fetchAggregated();
    const interval = setInterval(() => void fetchAggregated(), 30000);
    return () => clearInterval(interval);
  }, [handleSessionExpired]);

  // Fetch broker sources for account selector
  useEffect(() => {
    const fetchSources = async () => {
      try {
        const res = await fetch("/api/v1/portfolio/sources", { cache: "no-store" });
        if (res.status === 401) {
          handleSessionExpired();
          return;
        }
        if (res.ok) {
          const data = await res.json() as { id: string; name: string; broker_type: string; environment: string }[];
          setBrokerSources(data);
        }
      } catch {
        // silently skip
      }
    };
    void fetchSources();
  }, [handleSessionExpired]);

  const errorText = `${String((cockpitFetchError as Error | undefined)?.message ?? "")} ${String((metricsFetchError as Error | undefined)?.message ?? "")}`.toLowerCase();
  const authExpired = sessionExpiredRedirecting
    || errorText.includes("not authenticated")
    || errorText.includes("session expired")
    || errorText.includes("401");

  // ... effects ...
  useEffect(() => {
    if (authExpired) {
      handleSessionExpired();
    }
  }, [authExpired, handleSessionExpired]);

  useEffect(() => {
    setThemeMounted(true);
  }, []);

  const showLoading = (metricsLoading || cockpitLoading) && !mergedMetrics;
  const apiReachable = wsConnected || (cockpit?.status?.api_reachable ?? (!cockpitError || !metricsError));
  const stateFresh = wsConnected || (cockpit?.status?.state_fresh ?? Boolean(mergedMetrics?.status));
  const isDisconnected = !showLoading && !apiReachable;
  const isStale = apiReachable && !stateFresh;
  const mergedMetricRecord = (mergedMetrics ?? {}) as Record<string, unknown>;
  const mergedOpenPositions = sanitizeCount(wsData?.open_positions ?? cockpit?.status?.open_positions ?? cockpit?.positions?.length, 0);

  // Clean, non-duplicate calculation for options
  const _finalRawDerivs = cockpit?.derivatives || [];
  const _finalUniqueKeys = new Set(_finalRawDerivs.map((l: unknown) => {
    const leg = l as Record<string, unknown>;
    return `${leg.symbol}_${String(leg.expiry).replace(/-/g, "")}_${Number(leg.strike).toFixed(2)}_${leg.right}`;
  }));
  const mergedOptionPositions = _finalUniqueKeys.size > 0 ? _finalUniqueKeys.size : sanitizeCount(wsData?.option_positions ?? cockpit?.status?.option_positions, 0);

  const mergedOpenPositionsTotal = mergedOpenPositions + mergedOptionPositions;

  const sanitizedMetrics = useMemo(
    () =>
      sanitizeExecutionMetrics({
        ...(mergedMetrics ?? {}),
        open_positions: mergedOpenPositions,
        option_positions: mergedOptionPositions,
        open_positions_total: mergedOpenPositionsTotal,
      }) as Record<string, number>,
    [mergedMetrics, mergedOpenPositions, mergedOptionPositions, mergedOpenPositionsTotal],
  );
  const wsAggregatedEquity = sanitizeMoney(wsData?.aggregated_equity ?? wsData?.total_equity, Number.NaN);
  const capital = aggregatedEquity !== null ? aggregatedEquity : (Number.isFinite(wsAggregatedEquity) ? wsAggregatedEquity : sanitizedMetrics.capital);
  const dailyPnl = sanitizedMetrics.daily_pnl;
  const realizedPnl = Number.isFinite(Number((wsData as unknown as Record<string, unknown>)?.realized_pnl)) ? Number((wsData as unknown as Record<string, unknown>).realized_pnl) : 0;
  const activeMargin = Number.isFinite(Number((wsData as unknown as Record<string, unknown>)?.active_margin)) ? Number((wsData as unknown as Record<string, unknown>).active_margin) : 0;
  const leverageLimit = Number.isFinite(Number((wsData as unknown as Record<string, unknown>)?.leverage_limit)) ? Number((wsData as unknown as Record<string, unknown>).leverage_limit) : 1.0;
  const neuralLatencyMicros = useMemo(() => {
    const heatmap = (wsData as unknown as Record<string, unknown>)?.latency_heatmap as Array<{ name: string; micros: number }> | undefined;
    if (!heatmap || heatmap.length === 0) return null;
    return Math.max(...heatmap.map((e) => Number(e.micros) || 0));
  }, [wsData]);
  const dailyPnlByBroker = (cockpit?.status as unknown as Record<string, unknown>)?.daily_pnl_by_broker as Record<string, number> ?? { ibkr: 0, alpaca: 0 };
  const stressLiquidation = (cockpit?.status as unknown as Record<string, unknown>)?.stress_liquidation as Record<string, unknown> | null ?? null;
  const shadowDeployment = (cockpit?.status as unknown as Record<string, unknown>)?.shadow_deployment as Record<string, unknown> | null ?? null;
  const ibkrDailyPnl = sanitizeMoney(dailyPnlByBroker?.ibkr, 0);
  const alpacaDailyPnl = sanitizeMoney(dailyPnlByBroker?.alpaca, 0);
  // ibkrUnrealizedPnl and alpacaUnrealizedPnl computed from live positions via useMemo below
  const totalPnl = sanitizedMetrics.total_pnl;
  const sharpe = sanitizedMetrics.sharpe_ratio;
  const sortino = sanitizedMetrics.sortino_ratio ?? 0;
  const calmar = sanitizedMetrics.calmar_ratio ?? 0;
  const profitFactor = sanitizedMetrics.profit_factor ?? 0;
  const alphaRetention = sanitizedMetrics.alpha_retention ?? 0;
  const winRate = sanitizedMetrics.win_rate;
  const openPositions = sanitizedMetrics.open_positions;
  const tradesCount = sanitizedMetrics.trades_count;
  const drawdownPct = normalizeDrawdownPct(sanitizedMetrics.max_drawdown);


  // Parse WS positions to match CockpitPosition interface
  const cockpitPositions = useMemo(() => cockpit?.positions ?? [], [cockpit?.positions]);
  const cockpitPositionSourceHints = useMemo(() => {
    const hints = new Map<string, { source_id: string; broker_type: string }>();
    for (const position of cockpitPositions) {
      const symbol = String(position.symbol ?? "").trim().toUpperCase();
      if (!symbol) continue;
      const sideRaw = String(position.side ?? (position.qty < 0 ? "SHORT" : "LONG")).trim().toUpperCase();
      const side = sideRaw === "SHORT" ? "SHORT" : "LONG";
      const sourceId = String(position.source_id ?? "").trim();
      const brokerType = String(position.broker_type ?? "").trim().toLowerCase();
      if (!hints.has(`${symbol}|${side}`)) {
        hints.set(`${symbol}|${side}`, { source_id: sourceId, broker_type: brokerType });
      }
      if (!hints.has(symbol)) {
        hints.set(symbol, { source_id: sourceId, broker_type: brokerType });
      }
    }
    return hints;
  }, [cockpitPositions]);

  const wsPositions = useMemo(() => {
    if (!wsData?.positions) return null;
    return Object.entries(wsData.positions).map(([symbol, raw]) => {
      const data = (raw && typeof raw === "object") ? (raw as Record<string, unknown>) : {};
      const rawQty = sanitizeMoney(data.qty, 0);
      // Use trunc for equities (whole shares), but preserve fractional qty for crypto (e.g. 0.1234 BTC)
      const isCrypto = String(symbol).includes("/") || String(symbol).toUpperCase().startsWith("CRYPTO:");
      const qty = isCrypto ? rawQty : Math.trunc(rawQty);
      if (qty === 0) return null;
      const symbolUpper = String(symbol).trim().toUpperCase();
      const side = qty < 0 ? "SHORT" : "LONG";
      const sourceHint = cockpitPositionSourceHints.get(`${symbolUpper}|${side}`)
        ?? cockpitPositionSourceHints.get(symbolUpper);
      const sourceId = String(data.source_id ?? sourceHint?.source_id ?? "");
      const brokerType = String(data.broker_type ?? sourceHint?.broker_type ?? "");
      return {
        symbol: symbolUpper,
        qty,
        side,
        entry: sanitizeMoney(data.avg_price, 0),
        current: sanitizeMoney(data.current_price, 0),
        pnl: sanitizeMoney(data.pnl, 0),
        pnl_pct: sanitizeMoney(data.pnl_pct, 0),
        signal: sanitizeMoney(data.current_signal, 0),
        signal_direction: String(data.signal_direction ?? "UNKNOWN"),
        broker_type: brokerType,
        stale: Boolean(data.stale),
        source_status: String(data.source_status ?? ""),
        source_id: sourceId,
      };
    }).filter((row) => row !== null) as CockpitPosition[];
  }, [wsData, cockpitPositionSourceHints]);

  const positionsCandidate = useMemo(() => {
    if (wsPositions && wsPositions.length > 0) {
      return wsPositions;
    }
    if (cockpitPositions.length > 0) {
      return cockpitPositions;
    }
    return wsPositions ?? cockpitPositions;
  }, [wsPositions, cockpitPositions]);
  const positionsFeedDegraded = !apiReachable || Boolean(cockpitError) || Boolean(metricsError) || isStale;
  const confirmedFlatPositions = !positionsFeedDegraded
    && mergedOpenPositions === 0
    && (wsPositions?.length ?? 0) === 0
    && cockpitPositions.length === 0;
  useEffect(() => {
    if (positionsCandidate.length > 0) {
      setPositionsSnapshot(positionsCandidate);
      return;
    }
    if (confirmedFlatPositions) {
      setPositionsSnapshot([]);
    }
  }, [positionsCandidate, confirmedFlatPositions]);
  const positions = useMemo(() => {
    if (positionsCandidate.length > 0) {
      return positionsCandidate;
    }
    if (confirmedFlatPositions) {
      return [];
    }
    return positionsSnapshot;
  }, [positionsCandidate, confirmedFlatPositions, positionsSnapshot]);

  const { ibkrUnrealizedPnl, alpacaUnrealizedPnl } = useMemo(() => {
    let ibkr = 0, alpaca = 0;
    for (const p of positions) {
      const isCrypto = p.symbol.startsWith("CRYPTO:") || p.broker_type === "alpaca" || (p.source_id ?? "").toLowerCase().includes("alpaca");
      if (isCrypto) {
        alpaca += p.pnl;
      } else {
        ibkr += p.pnl;
      }
    }
    return { ibkrUnrealizedPnl: ibkr, alpacaUnrealizedPnl: alpaca };
  }, [positions]);

  const cockpitDerivatives = useMemo(() => cockpit?.derivatives ?? [], [cockpit?.derivatives]);
  const optionPositionsFromStatus = sanitizeCount(cockpit?.status?.option_positions, sanitizedMetrics.option_positions);
  const confirmedFlatDerivatives = !positionsFeedDegraded
    && optionPositionsFromStatus === 0
    && cockpitDerivatives.length === 0;
  useEffect(() => {
    if (cockpitDerivatives.length > 0) {
      setDerivativesSnapshot(cockpitDerivatives);
      return;
    }
    if (confirmedFlatDerivatives) {
      setDerivativesSnapshot([]);
    }
  }, [cockpitDerivatives, confirmedFlatDerivatives]);
  const derivatives = useMemo(() => {
    const rawList = cockpitDerivatives.length > 0
      ? cockpitDerivatives
      : (confirmedFlatDerivatives ? [] : derivativesSnapshot);

    // Deduplicate mismatched date formats (e.g., 2026-03-13 vs 20260313)
    const deduped = new Map<string, CockpitDerivative>();

    for (const leg of rawList) {
      // Standardize expiry to YYYYMMDD for matching
      const normExpiry = String(leg.expiry).replace(/-/g, "");
      const normStrike = Number(leg.strike).toFixed(2);
      const key = `${leg.symbol}_${normExpiry}_${normStrike}_${leg.right}`;

      if (deduped.has(key)) {
        const existing = deduped.get(key)!;
        // Prefer the clean dashed format for the UI display
        const displayExpiry = leg.expiry.includes("-") ? leg.expiry : (existing.expiry.includes("-") ? existing.expiry : leg.expiry);

        // Take the absolute highest quantity to prevent partial-sync ghost deflation
        const bestQty = Math.abs(leg.quantity) > Math.abs(existing.quantity) ? leg.quantity : existing.quantity;

        deduped.set(key, {
          ...existing,
          expiry: displayExpiry,
          quantity: bestQty
        });
      } else {
        deduped.set(key, { ...leg });
      }
    }

    return Array.from(deduped.values());
  }, [cockpitDerivatives, confirmedFlatDerivatives, derivativesSnapshot]);

  const cockpitSleeves = useMemo(() => cockpit?.attribution?.sleeves ?? [], [cockpit?.attribution?.sleeves]);
  const confirmedFlatSleeves = !positionsFeedDegraded
    && openPositions === 0
    && cockpitSleeves.length === 0;
  useEffect(() => {
    if (cockpitSleeves.length > 0) {
      setSleevesSnapshot(cockpitSleeves);
      return;
    }
    if (confirmedFlatSleeves) {
      setSleevesSnapshot([]);
    }
  }, [cockpitSleeves, confirmedFlatSleeves]);
  const sleeves = useMemo(() => {
    if (cockpitSleeves.length > 0) {
      return cockpitSleeves;
    }
    if (confirmedFlatSleeves) {
      return [];
    }
    return sleevesSnapshot;
  }, [cockpitSleeves, confirmedFlatSleeves, sleevesSnapshot]);

  const usp = cockpit?.status?.usp ?? null;
  const brokerRuntime = useMemo(() => {
    const rows = (Array.isArray(cockpit?.status?.brokers) ? cockpit!.status!.brokers : []) as unknown[];
    const byType = new Map(rows.map((row) => [ (row as Record<string, unknown>).broker, row]));

    const checkStatus = (b: any) => {
      if (!b) return { status: 'not_configured' as const, mode: 'idle' as const, source_count: 0, live_source_count: 0, heartbeat_ts: null, stale_age_seconds: null };
      return b;
    };

    return {
      alpaca: checkStatus(byType.get("alpaca")),
      ibkr: checkStatus(byType.get("ibkr")),
      active: cockpit?.status?.active_broker === "none" ? "MULTI" : (cockpit?.status?.active_broker ?? "MULTI"),
    };
  }, [cockpit?.status?.active_broker, cockpit?.status?.brokers]);

  const brokerStateToBadge = useCallback((
    state: "live" | "stale" | "configured" | "not_configured" | undefined,
    mode?: "trading" | "idle" | "disabled",
  ) => {
    if (mode === "trading" && state && state !== "not_configured") return "ok";
    if (mode === "idle") return "warn";
    if (state === "live") return "ok";
    if (state === "stale" || state === "configured") return "warn";
    return "down";
  }, []);

  const brokerDetail = useCallback((broker: {
    broker?: string;
    status: "live" | "stale" | "configured" | "not_configured";
    mode?: "trading" | "idle" | "disabled";
    source_count: number;
    live_source_count: number;
    heartbeat_ts?: string | null;
    stale_age_seconds?: number | null;
  } | undefined) => {
    if (!broker) return "not configured";
    const heartbeatAge = Number.isFinite(Number(broker.stale_age_seconds))
      ? `${Math.max(0, Math.trunc(Number(broker.stale_age_seconds)))}s`
      : "n/a";
    const heartbeatTime = broker.heartbeat_ts
      ? String(broker.heartbeat_ts).split("T")[1]?.replace(/\.\d+Z$/, "Z") ?? String(broker.heartbeat_ts)
      : "unknown";
    const heartbeatDetail = `${heartbeatAge} @ ${heartbeatTime}`;
    if (broker.mode === "idle") {
      const reason = broker.broker === "ibkr" ? "standby — equity market closed" : "idle (positions/metrics)";
      return `${reason} • hb ${heartbeatDetail}`;
    }
    if (broker.mode === "trading") {
      const label = broker.broker === "alpaca" ? "crypto 24/7 active" : "trading active";
      return `${label} • hb ${heartbeatDetail}`;
    }
    if (broker.status === "live") return `${broker.live_source_count}/${Math.max(1, broker.source_count)} live • hb ${heartbeatDetail}`;
    if (broker.status === "stale") return `configured (stale ${heartbeatDetail})`;
    if (broker.status === "configured") return `configured (idle) • hb ${heartbeatDetail}`;
    return "not configured";
  }, []);
  const optionPositions = optionPositionsFromStatus;
  const totalLines = openPositions + optionPositions;

  const cockpitStarting = Number(cockpit?.status?.starting_capital ?? 0);
  const startingCapital = sanitizedMetrics.starting_capital > 100
    ? sanitizedMetrics.starting_capital
    : (cockpitStarting > 100 ? cockpitStarting : Math.max(1, capital - totalPnl));
  const returnPct = startingCapital > 0 ? totalPnl / startingCapital : 0;
  const pnlPerTrade = tradesCount > 0 ? totalPnl / tradesCount : 0;
  const avgPositionSize = openPositions > 0 ? capital / openPositions : 0;
  const nowLabel = String(mergedMetrics?.timestamp ?? "--");

  const activeAlerts = useMemo(() => {
    const source = cockpit?.alerts ?? [];
    return source.filter((alert) => !dismissedAlerts[alert.id]);
  }, [cockpit?.alerts, dismissedAlerts]);

  const triageAlerts = useMemo(() => {
    if (alertFilter === "all") return activeAlerts;
    return activeAlerts.filter((alert) => alert.severity === alertFilter);
  }, [activeAlerts, alertFilter]);

  const sortedPositions = useMemo(() => {
    let copy = [...positions];
    if (selectedSourceId) {
      copy = copy.filter(p => p.source_id === selectedSourceId);
    }
    copy.sort((a, b) => {
      const cmp = comparePositions(a, b, sortKey);
      return sortDirection === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [positions, sortDirection, sortKey, selectedSourceId]);

  const lensModel = useMemo<Record<LensKey, LensModel>>(() => {
    return {
      performance: {
        title: "Performance Lens",
        subtitle: "Alpha quality, capital growth, and efficiency diagnostics.",
        rows: [
          {
            label: "Capital",
            value: formatCurrencyWithCents(capital),
            hint: "Live account equity",
          },
          {
            label: "Daily PnL",
            value: formatCurrency(dailyPnl),
            hint: "Combined intraday performance",
            tone: dailyPnl >= 0 ? "positive" : "negative",
          },
          {
            label: "IBKR PnL",
            value: formatCurrency(ibkrDailyPnl),
            hint: "Equities & FX (Interactive Brokers)",
            tone: ibkrDailyPnl >= 0 ? "positive" : "negative",
          },
          {
            label: "Crypto PnL",
            value: formatCurrency(alpacaDailyPnl),
            hint: "Crypto positions (Alpaca)",
            tone: alpacaDailyPnl >= 0 ? "positive" : "negative",
          },
          {
            label: "Net PnL",
            value: formatCurrency(totalPnl),
            hint: "Strategy total PnL",
            tone: totalPnl >= 0 ? "positive" : "negative",
          },
          {
            label: "Return",
            value: formatPct(returnPct),
            hint: "PnL over starting equity",
            tone: returnPct >= 0 ? "positive" : "negative",
          },
          {
            label: "Alpha Retention",
            value: `${(alphaRetention * 100).toFixed(1)}%`,
            hint: "Excess return vs SPY benchmark",
            tone: alphaRetention >= 0 ? "positive" : "negative",
          },
          {
            label: "Profit Factor",
            value: profitFactor === Infinity ? "∞" : profitFactor.toFixed(2),
            hint: "Gross profit / gross loss",
            tone: profitFactor >= 1.5 ? "positive" : profitFactor >= 1 ? "neutral" : "negative",
          },
        ],
        bars: [
          {
            label: "Sharpe Attainment",
            value: clampPct((sharpe / SHARPE_TARGET) * 100),
            targetLabel: `Target ${SHARPE_TARGET.toFixed(2)}`,
            tone: sharpe >= SHARPE_TARGET ? "positive" : "neutral",
          },
          {
            label: "Win Rate Quality",
            value: clampPct((winRate / WIN_RATE_TARGET) * 100),
            targetLabel: `Target ${(WIN_RATE_TARGET * 100).toFixed(0)}%`,
            tone: winRate >= WIN_RATE_TARGET ? "positive" : "neutral",
          },
          {
            label: "Capital Efficiency",
            value: clampPct((Math.abs(returnPct) / RETURN_CYCLE_TARGET) * 100),
            targetLabel: `${(RETURN_CYCLE_TARGET * 100).toFixed(0)}% cycle return`,
            tone: returnPct >= 0 ? "positive" : "negative",
          },
        ],
      },
      risk: {
        title: "Risk Lens",
        subtitle: "Drawdown stress, downside persistence, and control utilization.",
        rows: [
          {
            label: "Max Drawdown",
            value: `${drawdownPct.toFixed(2)}%`,
            hint: "Peak-to-trough equity draw",
            tone: drawdownPct > -8 ? "positive" : "negative",
          },
          {
            label: "Sharpe",
            value: sharpe.toFixed(2),
            hint: "Risk-adjusted return",
            tone: sharpe >= 1 ? "positive" : "negative",
          },
          {
            label: "Sortino",
            value: sortino.toFixed(2),
            hint: "Downside-risk-adjusted return",
            tone: sortino >= 1.5 ? "positive" : sortino >= 0 ? "neutral" : "negative",
          },
          {
            label: "Calmar",
            value: calmar.toFixed(2),
            hint: "Annual return / max drawdown",
            tone: calmar >= 1 ? "positive" : calmar >= 0 ? "neutral" : "negative",
          },
          {
            label: "Open Positions",
            value: String(openPositions),
            hint: "Current risk surface",
          },
          {
            label: "Trades",
            value: String(tradesCount),
            hint: "Execution sample size",
          },
        ],
        bars: [
          {
            label: "DD Budget Use",
            value: clampPct((Math.abs(drawdownPct) / DRAWDOWN_BUDGET_PCT) * 100),
            targetLabel: `${DRAWDOWN_BUDGET_PCT}% portfolio cap`,
            tone: Math.abs(drawdownPct) <= 8 ? "positive" : "negative",
          },
          {
            label: "Risk Buffer",
            value: clampPct(100 - (Math.abs(drawdownPct) / DRAWDOWN_BUDGET_PCT) * 100),
            targetLabel: "Higher is safer",
            tone: Math.abs(drawdownPct) <= 8 ? "positive" : "negative",
          },
          {
            label: "Governance Headroom",
            value: clampPct((sharpe / 2.0) * 100),
            targetLabel: "Sortino/Sharpe guard",
            tone: sharpe >= 1 ? "positive" : "neutral",
          },
        ],
      },
      execution: {
        title: "Execution Lens",
        subtitle: "Fill quality proxies and inventory footprint concentration.",
        rows: [
          {
            label: "Total Trades",
            value: String(tradesCount),
            hint: "Filled orders",
          },
          {
            label: "PnL / Trade",
            value: formatCurrency(pnlPerTrade),
            hint: "Average edge realization",
            tone: pnlPerTrade >= 0 ? "positive" : "negative",
          },
          {
            label: "Avg Position Size",
            value: formatCompactCurrency(avgPositionSize),
            hint: "Equity at risk per position",
          },
          {
            label: "Book Utilization",
            value: formatPct(openPositions / MAX_POSITIONS),
            hint: `Using ${MAX_POSITIONS}-slot max book`,
          },
        ],
        bars: [
          {
            label: "Execution Throughput",
            value: clampPct((tradesCount / TRADE_CYCLE_TARGET) * 100),
            targetLabel: `${TRADE_CYCLE_TARGET} trade cycle`,
            tone: tradesCount >= 20 ? "positive" : "neutral",
          },
          {
            label: "Position Dispersion",
            value: clampPct((openPositions / MAX_POSITIONS) * 100),
            targetLabel: "Concentration monitor",
            tone: openPositions <= Math.floor(MAX_POSITIONS * 0.6) ? "positive" : "negative",
          },
          {
            label: "Edge Capture",
            value: clampPct((pnlPerTrade / EDGE_CAPTURE_TARGET) * 100),
            targetLabel: `$${EDGE_CAPTURE_TARGET}/trade benchmark`,
            tone: pnlPerTrade >= 0 ? "positive" : "negative",
          },
        ],
      },
    };
  }, [capital, dailyPnl, ibkrDailyPnl, alpacaDailyPnl, drawdownPct, openPositions, pnlPerTrade, returnPct, sharpe, totalPnl, tradesCount, avgPositionSize, winRate]);

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  const setSort = (key: PositionSortKey) => {
    if (sortKey === key) {
      setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    setSortDirection("desc");
  };

  const dismissAlert = (id: string) => {
    setDismissedAlerts((prev) => ({ ...prev, [id]: true }));
  };

  const criticalCount = activeAlerts.filter((a) => a.severity === "critical").length;
  const warningCount = activeAlerts.filter((a) => a.severity === "warning").length;
  const uspScore = Number(usp?.score ?? 0);

  const readinessItems = useMemo(
    () => [
      {
        label: "API",
        state: isDisconnected ? "down" : isStale ? "warn" : "ok",
        detail: isDisconnected ? "unreachable" : isStale ? "stale feed" : "healthy",
      },
      {
        label: "Auth",
        state: authExpired ? "down" : "ok",
        detail: authExpired ? "session expired" : "session active",
      },
      {
        label: "USP Engine", state: Number(cockpit?.status?.usp_score ?? 0) > 70 ? "ok" : "warn",
        detail: !usp?.engine ? "missing scorecard" : `score ${uspScore.toFixed(1)}/100`,
      },
      {
        label: "Mandate Copilot", state: Number(cockpit?.status?.mandate_signals ?? 0) > 0 ? "ok" : "warn",
        detail: `${cockpit?.status?.mandate_signals ?? 0} signals`,
      },
      {
        label: "Workflow Pack", state: Number(cockpit?.status?.active_workflows ?? 0) > 0 ? "ok" : "warn",
        detail: `${cockpit?.status?.active_workflows ?? 0} active`,
      },
      {
        label: "Alpaca",
        state: brokerStateToBadge(brokerRuntime.alpaca?.status as any, brokerRuntime.alpaca?.mode as any),
        detail: brokerDetail(brokerRuntime.alpaca as any),
      },
      {
        label: "IBKR",
        state: brokerStateToBadge(brokerRuntime.ibkr?.status as any, brokerRuntime.ibkr?.mode as any),
        detail: brokerDetail(brokerRuntime.ibkr as any),
      },
    ],
    [
      authExpired,
      brokerDetail,
      brokerRuntime.alpaca,
      brokerRuntime.ibkr,
      brokerStateToBadge,
      cockpit?.status?.mandate_signals,
      cockpit?.status?.active_workflows,
      isDisconnected,
      isStale,
      usp?.engine,
      uspScore,
    ],
  );

  if (authExpired) {
    return (
      <main className="apex-shell min-h-screen px-4 py-6 sm:px-6 lg:px-10">
        <div className="mx-auto w-full max-w-3xl">
          <Alert className="rounded-2xl border-warning/40 bg-warning/10 text-warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Session expired</AlertTitle>
            <AlertDescription>Redirecting to login.</AlertDescription>
          </Alert>
        </div>
      </main>
    );
  }

  return (
    <main className="apex-shell min-h-screen px-4 py-6 sm:px-6 lg:px-10">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 lg:gap-8">
        <AlertsFeed>
          <div className="glass-card rounded-2xl p-4 transition-all duration-200">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex items-center gap-2 text-sm">
                <AlertTriangle className="h-4 w-4 text-primary" />
                <span className="font-semibold text-foreground">Alert Triage</span>
                <span className="rounded-full bg-negative/15 px-2 py-0.5 text-xs font-semibold text-negative">{criticalCount} critical</span>
                <span className="rounded-full bg-warning/15 px-2 py-0.5 text-xs font-semibold text-warning">{warningCount} warning</span>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {(["all", "critical", "warning", "info"] as SeverityFilter[]).map((filter) => (
                  <button
                    key={filter}
                    type="button"
                    aria-pressed={alertFilter === filter}
                    onClick={() => setAlertFilter(filter)}
                    className={`rounded-full px-2.5 py-1.5 text-xs font-semibold transition ${alertFilter === filter
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary text-secondary-foreground hover:bg-secondary/70"
                      }`}
                  >
                    {filter}
                  </button>
                ))}
                <Button size="sm" variant="outline" className="h-9 rounded-full px-3 text-xs" onClick={() => setDrawerOpen((v) => !v)}>
                  {drawerOpen ? "Hide" : "Show"}
                </Button>
              </div>
            </div>

            {drawerOpen ? (
              <div className="mt-3 space-y-2">
                {triageAlerts.length === 0 ? (
                  <p className="rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
                    No active alerts for this filter.
                  </p>
                ) : (
                  triageAlerts.slice(0, 5).map((alert) => (
                    <div key={alert.id} className="flex flex-col gap-2 rounded-lg border border-border/70 bg-background/60 px-3 py-2 sm:flex-row sm:items-start sm:justify-between">
                      <div className="space-y-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className={`rounded-full px-2 py-0.5 text-xs font-semibold uppercase ${severityBadgeClass(alert.severity)}`}>
                            {alert.severity}
                          </span>
                          <span className="text-xs uppercase tracking-wide text-muted-foreground">{alert.source}</span>
                          <p className="text-sm font-semibold text-foreground">{alert.title}</p>
                        </div>
                        <p className="text-xs text-muted-foreground">{alert.detail}</p>
                      </div>
                      <Button size="sm" variant="outline" className="h-9 rounded-full px-3 text-xs" onClick={() => dismissAlert(alert.id)}>
                        Dismiss
                      </Button>
                    </div>
                  ))
                )}
              </div>
            ) : null}
          </div>
        </AlertsFeed>

        <header className="glass-card apex-fade-up rounded-3xl p-5 sm:p-6">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
            <div className="space-y-2">
              <p className="inline-flex items-center gap-2 rounded-full bg-secondary px-3 py-1 text-xs font-semibold tracking-wide text-secondary-foreground">
                <ShieldCheck className="h-3.5 w-3.5" />
                APEX Trading System
              </p>
              <h1 className="text-4xl font-bold tracking-tight text-foreground">Apex Terminal</h1>
              <p className="text-sm font-medium text-muted-foreground mt-1">Institutional Execution Monitoring</p>
              <p className="text-xs text-muted-foreground/60">Desk Sync: {nowLabel}</p>
              {aggregatedEquity !== null && (
                <p className="inline-flex items-center gap-1.5 text-xs font-medium text-positive">
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-positive" />
                  Total Equity ({brokerCount} broker{brokerCount !== 1 ? "s" : ""}): {formatCurrency(aggregatedEquity)}
                </p>
              )}
            </div>

            <ControlsPanel>
              <span
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${isDisconnected
                  ? "bg-destructive/15 text-destructive"
                  : isStale
                    ? "bg-warning/15 text-warning"
                    : "bg-positive/15 text-positive"
                  }`}
              >
                {isDisconnected ? <WifiOff className="h-3.5 w-3.5" /> : <Wifi className="h-3.5 w-3.5" />}
                {isDisconnected ? "Disconnected" : isStale ? "Connected (Stale)" : "Connected"}
              </span>
              {/* Broker mode indicator — always visible in header */}
              <button
                type="button"
                onClick={() => setActiveTab("trading")}
                title="Click to manage broker routing"
                className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold transition hover:opacity-80 ${
                  ((wsData as any)?.broker_mode ?? "both") === "alpaca"
                    ? "bg-positive/15 text-positive"
                    : "bg-primary/15 text-primary"
                }`}
              >
                <Server className="h-3.5 w-3.5" />
                {((wsData as Record<string, unknown>)?.broker_mode ?? "both") === "alpaca" ? "Alpaca Only" : "IBKR Split"}
              </button>
              <Button variant="outline" className="rounded-xl" onClick={() => window.location.reload()}>
                <RefreshCw className="h-4 w-4" />
                Refresh
              </Button>
              <Button
                variant="outline"
                className="rounded-xl"
                onClick={toggleTheme}
                aria-label={
                  !themeMounted
                    ? "Toggle theme"
                    : theme === "dark"
                      ? "Switch to light mode"
                      : "Switch to dark mode"
                }
              >
                {!themeMounted ? <Moon className="h-4 w-4" /> : theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                {!themeMounted ? "Theme" : theme === "dark" ? "Light" : "Dark"}
              </Button>
              {isPublic ? null : (
                <>
                  <Button variant="outline" className="rounded-xl" onClick={handleLogout}>
                    <LogOut className="h-4 w-4" />
                    Logout
                  </Button>
                  <Link
                    href="/settings"
                    className="inline-flex h-9 items-center justify-center rounded-xl border border-border bg-background px-3 text-sm font-medium text-foreground transition hover:bg-secondary/50"
                  >
                    Settings
                  </Link>
                  <Link
                    href="/pricing"
                    className="inline-flex h-9 items-center justify-center rounded-xl border border-border bg-background px-3 text-sm font-medium text-foreground transition hover:bg-secondary/50"
                  >
                    Pricing
                  </Link>
                </>
              )}
            </ControlsPanel>
          </div>
        </header>

        {isDisconnected ? (
          <Alert variant="destructive" className="apex-fade-up rounded-2xl">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Disconnected</AlertTitle>
            <AlertDescription>Backend service is unreachable. Displaying cached data if available.</AlertDescription>
          </Alert>
        ) : null}

        {isStale ? (
          <Alert className="apex-fade-up rounded-2xl border-warning/40 bg-warning/10 text-warning">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>State stale</AlertTitle>
            <AlertDescription>Backend is reachable, but trading state freshness is below threshold.</AlertDescription>
          </Alert>
        ) : null}

        {!wsConnected && (reconnectAttempt ?? 0) > 0 ? (
          <Alert className="apex-fade-up rounded-2xl border-warning/40 bg-warning/10 text-warning">
            <RefreshCw className="h-4 w-4 animate-spin" />
            <AlertTitle>Reconnecting</AlertTitle>
            <AlertDescription>WebSocket attempt #{reconnectAttempt} — auto-retrying with backoff (max 30 s).</AlertDescription>
          </Alert>
        ) : null}

        <section className="glass-card apex-fade-up rounded-2xl p-4 transition-all duration-200">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-foreground">Function Readiness</h2>
            <span className="text-xs text-muted-foreground">
              Core + USP operating signals {brokerRuntime.active !== "none" ? `• active broker: ${String(brokerRuntime.active).toUpperCase()}` : ""}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-7">
            {readinessItems.map((item) => (
              <div key={item.label} className="rounded-xl border border-border/70 bg-background/60 px-3 py-2">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{item.label}</p>
                  <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase ${readinessClass(item.state as "ok" | "warn" | "down")}`}>
                    {item.state}
                  </span>
                </div>
                <p className="text-xs text-foreground">{item.detail}</p>
              </div>
            ))}
          </div>
        </section>

        <EquityPanel>
          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Capital</p>
              <DollarSign className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-20" /> : formatCurrencyWithCents(capital)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Live equity base</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Daily PnL</p>
              {dailyPnl >= 0 ? <TrendingUp className="h-4 w-4 text-positive" /> : <TrendingDown className="h-4 w-4 text-negative" />}
            </div>
            <p className={`apex-kpi-value mt-2 text-lg font-semibold ${dailyPnl >= 0 ? "text-positive" : "text-negative"}`}>
              {showLoading ? <Skeleton className="h-5 w-20" /> : formatCompactCurrency(dailyPnl)}
            </p>
            <div className="mt-1.5 flex items-center gap-3 text-[11px]">
              <span className={`font-medium ${ibkrDailyPnl >= 0 ? "text-positive" : "text-negative"}`}
                title={`IBKR unrealized: ${formatCompactCurrency(ibkrUnrealizedPnl)}`}>
                IBKR {formatCompactCurrency(ibkrDailyPnl)}
                {ibkrUnrealizedPnl !== 0 && (
                  <span className="ml-1 opacity-60 text-[10px]">({formatCompactCurrency(ibkrUnrealizedPnl)} unrlzd)</span>
                )}
              </span>
              <span className="text-border">|</span>
              <span className={`font-medium ${alpacaDailyPnl >= 0 ? "text-positive" : "text-negative"}`}
                title={`Crypto unrealized: ${formatCompactCurrency(alpacaUnrealizedPnl)}`}>
                Crypto {formatCompactCurrency(alpacaDailyPnl)}
                {alpacaUnrealizedPnl !== 0 && (
                  <span className="ml-1 opacity-60 text-[10px]">({formatCompactCurrency(alpacaUnrealizedPnl)} unrlzd)</span>
                )}
              </span>
            </div>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Sharpe</p>
              <Gauge className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-14" /> : sharpe.toFixed(2)}</p>
            <p className="mt-1 text-xs text-muted-foreground">{`Target ${SHARPE_TARGET.toFixed(2)}+`}</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("risk")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Max Drawdown</p>
              <AlertTriangle className="h-4 w-4 text-primary" />
            </div>
            <p className={`apex-kpi-value mt-2 text-lg font-semibold ${drawdownPct > -8 ? "text-positive" : "text-negative"}`}>
              {showLoading ? <Skeleton className="h-5 w-16" /> : `${drawdownPct.toFixed(2)}%`}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">{`Budget under ${DRAWDOWN_BUDGET_PCT}%`}</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("execution")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Open Positions</p>
              <BarChart3 className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-10" /> : String(openPositions)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Equity symbols only</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("execution")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Total Trades</p>
              <Timer className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-10" /> : String(tradesCount)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Execution sample</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">USP Engine</p>
              <ShieldCheck className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-16" /> : `${Number(usp?.score ?? 0).toFixed(1)}/100`}</p>
            <p className="mt-1 text-xs capitalize text-muted-foreground">
              {showLoading ? <Skeleton className="h-3 w-20" /> : String(usp?.band ?? "stabilize").replaceAll("_", " ")}
            </p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("risk")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Sortino</p>
              <Gauge className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-14" /> : sortino.toFixed(2)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Downside adjusted</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Profit Factor</p>
              <TrendingUp className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-14" /> : profitFactor.toFixed(2)}</p>
            <p className="mt-1 text-xs text-muted-foreground">Gross PnL Ratio</p>
          </button>

          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Alpha Retention</p>
              <BarChart3 className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">{showLoading ? <Skeleton className="h-5 w-14" /> : `${(alphaRetention * 100).toFixed(2)}%`}</p>
            <p className="mt-1 text-xs text-muted-foreground">vs SPY Benchmark</p>
          </button>

          {/* Today Realized PnL */}
          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("performance")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Realized PnL</p>
              {realizedPnl >= 0 ? <TrendingUp className="h-4 w-4 text-positive" /> : <TrendingDown className="h-4 w-4 text-negative" />}
            </div>
            <p className={`apex-kpi-value mt-2 text-lg font-semibold ${realizedPnl >= 0 ? "text-positive" : "text-negative"}`}>
              {showLoading ? <Skeleton className="h-5 w-20" /> : formatCompactCurrency(realizedPnl)}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">Closed trades only</p>
          </button>

          {/* Active Margin */}
          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("risk")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Active Margin</p>
              <ShieldCheck className="h-4 w-4 text-primary" />
            </div>
            <p className="apex-kpi-value mt-2 text-lg font-semibold text-foreground">
              {showLoading ? <Skeleton className="h-5 w-20" /> : formatCompactCurrency(activeMargin)}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">Required margin notional</p>
          </button>

          {/* Risk Allocation (leverage limit) */}
          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("risk")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Risk Allocation</p>
              <Gauge className="h-4 w-4 text-primary" />
            </div>
            <p className={`apex-kpi-value mt-2 text-lg font-semibold ${leverageLimit >= 1.0 ? "text-positive" : leverageLimit >= 0.5 ? "text-foreground" : "text-negative"}`}>
              {showLoading ? <Skeleton className="h-5 w-14" /> : `${(leverageLimit * 100).toFixed(0)}%`}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">Global leverage gate</p>
          </button>

          {/* Neural Latency */}
          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left" onClick={() => setActiveLens("execution")}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Neural Latency</p>
              <Timer className="h-4 w-4 text-primary" />
            </div>
            <p className={`apex-kpi-value mt-2 text-lg font-semibold ${neuralLatencyMicros !== null && neuralLatencyMicros < 2000 ? "text-positive" : neuralLatencyMicros !== null ? "text-warning" : "text-muted-foreground"}`}>
              {showLoading ? <Skeleton className="h-5 w-14" /> : neuralLatencyMicros !== null ? `${neuralLatencyMicros.toFixed(0)} µs` : "—"}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">Max strategy latency</p>
          </button>

          {/* Equity Sparkline (mini) */}
          <button type="button" className="apex-panel apex-interactive rounded-2xl p-4 text-left col-span-1 xl:col-span-2" onClick={() => setActiveTab("equitycurve")}>
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Equity Curve</p>
              <TrendingUp className="h-4 w-4 text-primary" />
            </div>
            {showLoading ? (
              <Skeleton className="h-12 w-full" />
            ) : (
              <svg viewBox="0 0 200 48" className="w-full h-12" preserveAspectRatio="none">
                <polyline
                  points={(() => {
                    const eq = capital;
                    const start = Math.max(1, eq - Math.abs(dailyPnl) * 10);
                    const pts = Array.from({ length: 20 }, (_, i) => {
                      const x = (i / 19) * 200;
                      const progress = i / 19;
                      const y = 48 - ((start + (eq - start) * progress - start) / Math.max(1, eq - start)) * 44 - 2;
                      return `${x.toFixed(1)},${Math.max(2, Math.min(46, y)).toFixed(1)}`;
                    });
                    return pts.join(" ");
                  })()}
                  fill="none"
                  stroke={dailyPnl >= 0 ? "hsl(var(--positive))" : "hsl(var(--negative))"}
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            )}
            <p className="mt-1 text-xs text-muted-foreground">Today trend · click for full curve</p>
          </button>
        </EquityPanel>

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[1.25fr_0.75fr]">
          <PositionsTable>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-foreground">KPI Drilldown</h2>
                <p className="text-sm text-muted-foreground">{lensModel[activeLens].subtitle}</p>
              </div>
              <div className="apex-segment" role="tablist" aria-label="KPI lens selector">
                {(["performance", "risk", "execution"] as LensKey[]).map((lens) => (
                  <button
                    key={lens}
                    role="tab"
                    aria-selected={activeLens === lens}
                    type="button"
                    onClick={() => setActiveLens(lens)}
                    className={`apex-segment-button ${activeLens === lens ? "is-active" : ""}`}
                  >
                    {lens}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-5 grid gap-4 lg:grid-cols-2">
              <div className="space-y-3">
                <p className="text-sm font-semibold text-foreground">{lensModel[activeLens].title}</p>
                {lensModel[activeLens].rows.map((row) => (
                  <div key={row.label} className="rounded-xl border border-border/80 bg-background/70 p-3">
                    <div className="flex items-center justify-between gap-4">
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">{row.label}</p>
                      <p className={`apex-kpi-value text-sm font-semibold ${toneClass(row.tone)}`}>{showLoading ? <Skeleton className="h-4 w-16" /> : row.value}</p>
                    </div>
                    <p className="mt-1 text-[11px] text-muted-foreground">{row.hint}</p>
                  </div>
                ))}
              </div>

              <div className="space-y-3">
                <p className="text-sm font-semibold text-foreground">Control Gauges</p>
                {lensModel[activeLens].bars.map((bar) => (
                  <div key={bar.label} className="rounded-xl border border-border/80 bg-background/70 p-3">
                    <div className="mb-2 flex items-center justify-between gap-4 text-xs">
                      <span className="font-medium text-foreground">{bar.label}</span>
                      <span className="text-muted-foreground">{bar.targetLabel}</span>
                    </div>
                    <div className="apex-progress-track">
                      <div className={`apex-progress-fill ${barToneClass(bar.tone)}`} style={{ width: `${showLoading ? 0 : clampPct(bar.value)}%` }} />
                    </div>
                    <p className="apex-kpi-value mt-1 text-[11px] font-medium text-muted-foreground">{showLoading ? <Skeleton className="h-3 w-10" /> : `${clampPct(bar.value).toFixed(0)}%`}</p>
                  </div>
                ))}
              </div>
            </div>
          </PositionsTable>

          <article className="apex-panel apex-fade-up rounded-2xl p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-foreground">Desk Summary</h2>
              <Gauge className="h-4 w-4 text-primary" />
            </div>

            <dl className="mt-4 space-y-2 text-sm">
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Session status</dt>
                <dd className="font-medium text-foreground">
                  {isDisconnected ? "Disconnected" : isStale ? "Stale feed" : "Active"}
                </dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Win rate</dt>
                <dd className="apex-kpi-value font-medium text-foreground">{showLoading ? <Skeleton className="h-4 w-14" /> : formatPct(winRate)}</dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Net PnL</dt>
                <dd className={`apex-kpi-value font-medium ${totalPnl >= 0 ? "text-positive" : "text-negative"}`}>
                  {showLoading ? <Skeleton className="h-4 w-16" /> : formatCompactCurrency(totalPnl)}
                </dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Unrealized IBKR</dt>
                <dd className={`apex-kpi-value font-medium ${ibkrUnrealizedPnl >= 0 ? "text-positive" : "text-negative"}`}>
                  {showLoading ? <Skeleton className="h-4 w-16" /> : formatCompactCurrency(ibkrUnrealizedPnl)}
                </dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Unrealized Crypto</dt>
                <dd className={`apex-kpi-value font-medium ${alpacaUnrealizedPnl >= 0 ? "text-positive" : "text-negative"}`}>
                  {showLoading ? <Skeleton className="h-4 w-16" /> : formatCompactCurrency(alpacaUnrealizedPnl)}
                </dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Book utilization</dt>
                <dd className="apex-kpi-value font-medium text-foreground">{showLoading ? <Skeleton className="h-4 w-14" /> : formatPct(openPositions / MAX_POSITIONS)}</dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Alpha retention</dt>
                <dd className="apex-kpi-value font-medium text-foreground">
                  {showLoading ? <Skeleton className="h-4 w-14" /> : `${(alphaRetention * 100).toFixed(1)}%`}
                </dd>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <dt className="text-muted-foreground">Data cadence</dt>
                <dd className="font-medium text-foreground">5s polling</dd>
              </div>
            </dl>
          </article>
        </section>

        {/* --- EXPLAINABLE AI SHAP CHART INJECTION --- */}
        <section className="apex-fade-up mb-4 w-full">
          <ExplainableAIChart shapData={activeShapData} symbol="System Aggregate" />
        </section>

        {/* ── Tab Navigation ── */}
        <nav className="flex items-center gap-2 rounded-2xl border border-border/70 bg-background/60 p-1.5">
          {([
            ["missioncontrol", "Mission Control"],
            ["trading", "Trading"],
            ["mandate", "Mandate Copilot"],
            ["social", "Social Audit"],
            ["walkforward", "Walk-Forward"],
            ["heat", "Portfolio Heat"],
            ["diagnostics", "Gate Diagnostics"],
            ["factorpnl", "Factor P&L"],
            ["corrmap", "Correlation"],
            ["regimebt", "Regime Backtest"],
            ["wssmetrics", "WSS Metrics"],
            ["icreport", "IC / Decay"],
            ["postmortem", "Post-Mortem"],
            ["equitycurve", "Equity Curve"],
            ["stress", "Stress Engine"],
            ["attribution", "Attribution"],
            ["missed", "Missed Opps"],
            ["advancedmetrics", "Risk Metrics"],
            ["tca", "TCA Report"],
            ["stressscenarios", "Stress Test"],
            ["orderrejections", "Rejections"],
            ["featureic", "Feature IC"],
            ["abgate", "A/B Gate"],
            ["portfolioweights", "Portfolio Weights"],
            ["hmmregime", "HMM Regime"],
            ["alerts", "Alerts"],
            ["paperaccount", "Paper Account"],
            ["modelregistry", "Model Registry"],
            ["crossassetpairs", "Cross-Asset Pairs"],
            ["briefings", "Briefings Archive"],
            ["ivcrush", "IV Crush"],
            ["brokersync", "Broker Sync"],
          ] as [DashboardTab, string][]).map(([tab, label]) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${activeTab === tab
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                }`}
            >
              {label}
            </button>
          ))}
        </nav>

        {activeTab === "missioncontrol" ? (
          <MissionControlPanel />
        ) : activeTab === "mandate" ? (
          <MandateCopilotPanel onSessionExpired={handleSessionExpired} />
        ) : activeTab === "social" ? (
          <SocialGovernorPanel socialAudit={cockpit?.status?.social_audit as SocialGovernorPanelProps["socialAudit"]} />
        ) : activeTab === "walkforward" ? (
          <WalkForwardPanel />
        ) : activeTab === "heat" ? (
          <PortfolioHeatPanel />
        ) : activeTab === "diagnostics" ? (
          <RegimeSharpePanel />
        ) : activeTab === "factorpnl" ? (
          <FactorPnlPanel />
        ) : activeTab === "corrmap" ? (
          <CorrelationHeatmapPanel />
        ) : activeTab === "regimebt" ? (
          <RegimeBacktestPanel />
        ) : activeTab === "wssmetrics" ? (
          <div className="p-4 max-w-sm">
            <WssMetricsWidget />
          </div>
        ) : activeTab === "icreport" ? (
          <ICReportPanel />
        ) : activeTab === "postmortem" ? (
          <TradePostMortemPanel />
        ) : activeTab === "equitycurve" ? (
          <EquityCurvePanel />
        ) : activeTab === "stress" ? (
          <StressPanel />
        ) : activeTab === "attribution" ? (
          <AttributionPanel />
        ) : activeTab === "missed" ? (
          <MissedOpportunitiesPanel />
        ) : activeTab === "advancedmetrics" ? (
          <AdvancedMetricsPanel />
        ) : activeTab === "tca" ? (
          <TcaReportPanel />
        ) : activeTab === "stressscenarios" ? (
          <StressScenariosPanel />
        ) : activeTab === "orderrejections" ? (
          <OrderRejectionsPanel />
        ) : activeTab === "featureic" ? (
          <FeatureIcPanel />
        ) : activeTab === "abgate" ? (
          <AbGatePanel />
        ) : activeTab === "portfolioweights" ? (
          <PortfolioWeightsPanel />
        ) : activeTab === "hmmregime" ? (
          <HmmRegimePanel />
        ) : activeTab === "alerts" ? (
          <AlertHistoryPanel />
        ) : activeTab === "paperaccount" ? (
          <PaperAccountPanel />
        ) : activeTab === "modelregistry" ? (
          <ModelRegistryPanel />
        ) : activeTab === "crossassetpairs" ? (
          <CrossAssetPairsPanel />
        ) : activeTab === "briefings" ? (
          <BriefingArchivePanel />
        ) : activeTab === "ivcrush" ? (
          <IVCrushPanel />
        ) : activeTab === "brokersync" ? (
          <div className="apex-panel apex-fade-up rounded-2xl p-5">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-foreground">Broker Reconciliation</h2>
                <p className="text-sm text-muted-foreground">
                  All raw Alpaca positions vs active APEX strategy pairs — orphaned positions highlighted in red.
                </p>
              </div>
              <span className="rounded-full bg-negative/15 px-2.5 py-1 text-xs font-semibold text-negative">
                Live · 30 s refresh
              </span>
            </div>
            <BrokerReconciliationPanel
              brokerPositions={(wsData as unknown as Record<string, unknown>)?.broker_positions as BrokerPosition[]}
              lastUpdated={(wsData as unknown as Record<string, unknown>)?.timestamp as string}
            />
          </div>
        ) : null}


        {activeTab === "trading" ? (
          <>
            {/* Broker Routing — full-width, top of trading tab */}
            <BrokerModePanel brokerMode={(wsData as unknown as Record<string, unknown>)?.broker_mode as string ?? "both"} />

            <section className="grid grid-cols-1 gap-4 xl:grid-cols-[0.95fr_1.05fr]">
              <article className="apex-panel apex-fade-up rounded-2xl p-5">
                <div className="mb-4 flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-foreground">Sleeve Attribution</h2>
                  <span className="text-xs text-muted-foreground">30d lookback</span>
                </div>

                <div className="lg:col-span-1">
                  <SentimentHealthPanel
                    sentiment_health={(wsData as unknown as Record<string, unknown>)?.sentiment_health as Record<string, VetoDetail>}
                  />
                  <div className="mt-4">
                     <StrategyAllocationPanel allocation={(wsData as unknown as Record<string, unknown>)?.strategy_allocation as StrategyAllocationData} />
                  </div>
                  <div className="mt-4">
                     <SocialPulsePanel pulse={(wsData as unknown as Record<string, unknown>)?.social_pulse as SocialPulseData} />
                  </div>
                </div>
                <div className="lg:col-span-2">
                  <ControlsPanel />
                </div>

                <div className="space-y-3">
                  {sleeves.length === 0 ? (
                    <p className="rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-xs text-muted-foreground">No sleeve attribution yet.</p>
                  ) : (
                    sleeves.map((sleeve: SleeveAttribution) => (
                      <div key={sleeve.sleeve} className="rounded-xl border border-border/80 bg-background/70 p-3">
                        <div className="flex items-center justify-between gap-3">
                          <p className="text-sm font-semibold capitalize text-foreground">{formatSleeveLabel(sleeve.sleeve)}</p>
                          <span className="text-xs text-muted-foreground">{sleeve.trades} trades</span>
                        </div>
                        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                          <div className="rounded-lg border border-border/70 bg-background/60 px-2 py-1.5">
                            <p className="text-muted-foreground">Net alpha</p>
                            <p className={`apex-kpi-value mt-0.5 text-sm font-semibold ${sleeve.net_pnl >= 0 ? "text-positive" : "text-negative"}`}>
                              {formatCurrencyWithCents(sleeve.net_pnl)}
                            </p>
                          </div>
                          <div className="rounded-lg border border-border/70 bg-background/60 px-2 py-1.5">
                            <p className="text-muted-foreground">Execution drag</p>
                            <p className="apex-kpi-value mt-0.5 text-sm font-semibold text-foreground">{formatCurrencyWithCents(sleeve.modeled_execution_drag)}</p>
                          </div>
                          <div className="rounded-lg border border-border/70 bg-background/60 px-2 py-1.5">
                            <p className="text-muted-foreground">Slippage drag</p>
                            <p className="apex-kpi-value mt-0.5 text-sm font-semibold text-foreground">{formatCurrencyWithCents(sleeve.modeled_slippage_drag)}</p>
                          </div>
                          <div className="rounded-lg border border-border/70 bg-background/60 px-2 py-1.5">
                            <p className="text-muted-foreground">Drag / gross</p>
                            <p className="apex-kpi-value mt-0.5 text-sm font-semibold text-foreground">{formatPct(sleeve.execution_drag_pct_of_gross || 0)}</p>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </article>

              <PositionsTable>
                <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <h2 className="text-lg font-semibold text-foreground">Position Book (Sortable)</h2>
                  <div className="flex flex-wrap items-center gap-2">
                    {!isPublic && brokerSources.length > 0 && (
                      <select
                        id="account-selector"
                        value={selectedSourceId ?? ""}
                        onChange={(e) => setSelectedSourceId(e.target.value || null)}
                        className="h-8 rounded-lg border border-border bg-background px-2 text-xs font-medium text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                        aria-label="Filter by account"
                      >
                        <option value="">All Accounts</option>
                        {brokerSources.map((src) => (
                          <option key={src.id} value={src.id}>
                            {src.name} ({src.environment})
                          </option>
                        ))}
                      </select>
                    )}
                    <span className="text-xs text-muted-foreground">
                      {openPositions} equity + {optionPositions} options = {totalLines} lines
                    </span>
                  </div>
                </div>


                {selectedSourceId && brokerSources.length > 0 && (() => {
                  const src = brokerSources.find(s => s.id === selectedSourceId);
                  return src ? (
                    <p className="mb-2 rounded-lg border border-positive/30 bg-positive/10 px-3 py-2 text-xs font-medium text-positive">
                      Viewing: <strong>{src.name}</strong> ({src.broker_type.toUpperCase()} · {src.environment}) - positions scope active
                    </p>
                  ) : null;
                })()}

                {stressLiquidation?.active ? (
                  <div className="mb-3 rounded-xl border border-warning/30 bg-warning/10 p-3">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <p className="text-sm font-semibold text-foreground">Stress Liquidation In Progress</p>
                        <p className="text-xs text-muted-foreground">
                          {Number(stressLiquidation.completed_count ?? 0)} completed · {Number(stressLiquidation.in_progress_count ?? 0)} in progress · {Number(stressLiquidation.candidate_count ?? 0)} planned
                        </p>
                        {stressLiquidation.plan_id ? (
                          <p className="mt-1 text-xs text-muted-foreground">
                            Plan {String(stressLiquidation.plan_id)}
                            {Number(stressLiquidation.plan_epoch ?? 0) > 0 ? ` · epoch ${Number(stressLiquidation.plan_epoch)}` : ""}
                          </p>
                        ) : null}
                      </div>
                      <div className="flex items-center gap-2">
                        <p className="text-xs text-muted-foreground">
                          Active de-risking is being routed through the audited exit path.
                        </p>
                        {stressLiquidation.plan_id && stressLiquidation.plan_audit_url ? (
                          <Button
                            type="button"
                            size="xs"
                            variant={replayTarget?.kind === "plan" && replayTarget?.planId === stressLiquidation.plan_id ? "secondary" : "outline"}
                            onClick={() =>
                              setReplayTarget({
                                kind: "plan",
                                label: `Plan ${String(stressLiquidation.plan_id)}`,
                                requestUrl: String(stressLiquidation.plan_audit_url),
                                planId: String(stressLiquidation.plan_id),
                              })
                            }
                          >
                            Audit Plan
                          </Button>
                        ) : null}
                      </div>
                    </div>
                    {Array.isArray(stressLiquidation.candidates) && stressLiquidation.candidates.length > 0 ? (
                      <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                        {stressLiquidation.candidates.map((candidate) => (
                          <div key={String(candidate.symbol)} className="rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-xs">
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-semibold text-foreground">{String(candidate.symbol)}</span>
                              <span className="uppercase text-muted-foreground">{String(candidate.status || "pending").replaceAll("_", " ")}</span>
                            </div>
                            <p className="mt-1 text-muted-foreground">
                              {String(candidate.action || "partial_reduce") === "full_exit"
                                ? "100% exit"
                                : `${(Number(candidate.target_reduction_pct || 0) * 100).toFixed(0)}% cut`}
                              {" · "}
                              {(Number(candidate.progress_pct || 0) * 100).toFixed(0)}% complete
                            </p>
                            <p className="mt-1 text-muted-foreground">
                              {Number(candidate.executed_reduction_qty || 0).toFixed(2)} / {Number(candidate.planned_reduction_qty || 0).toFixed(2)} reduced
                              {" · "}
                              live qty {Number(candidate.live_qty || 0).toFixed(2)}
                            </p>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {shadowDeployment?.candidate_id ? (
                  <div className={`mb-3 rounded-xl border p-3 ${
                    shadowDeployment.status === "blocked"
                      ? "border-negative/30 bg-negative/10"
                      : shadowDeployment.status === "staged"
                        ? "border-primary/30 bg-primary/10"
                        : "border-positive/30 bg-positive/10"
                  }`}>
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <p className="text-sm font-semibold text-foreground">Shadow Deployment Gate</p>
                        <p className="text-xs text-muted-foreground">
                          {String(shadowDeployment.candidate_id)}
                          {shadowDeployment.model_version ? ` · model ${String(shadowDeployment.model_version)}` : ""}
                          {shadowDeployment.governor_policy_version ? ` · policy ${String(shadowDeployment.governor_policy_version)}` : ""}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Status</p>
                        <p className="text-sm font-semibold text-foreground">{String(shadowDeployment.status || "inactive").replaceAll("_", " ")}</p>
                      </div>
                    </div>
                    <div className="mt-3 grid gap-2 md:grid-cols-4">
                      <div className="rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-xs">
                        <p className="text-muted-foreground">Observed signals</p>
                        <p className="font-medium text-foreground">{Number(shadowDeployment.observed_signals || 0)}</p>
                      </div>
                      <div className="rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-xs">
                        <p className="text-muted-foreground">Agreement</p>
                        <p className="font-medium text-foreground">{(Number(shadowDeployment.decision_agreement_rate || 0) * 100).toFixed(0)}%</p>
                      </div>
                      <div className="rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-xs">
                        <p className="text-muted-foreground">Offline Sharpe</p>
                        <p className="font-medium text-foreground">{Number(shadowDeployment.offline_sharpe || 0).toFixed(2)}</p>
                      </div>
                      <div className="rounded-lg border border-border/60 bg-background/60 px-3 py-2 text-xs">
                        <p className="text-muted-foreground">Live Sharpe</p>
                        <p className="font-medium text-foreground">{Number(shadowDeployment.live_sharpe || 0).toFixed(2)}</p>
                      </div>
                    </div>
                    {Array.isArray(shadowDeployment.reasons) && shadowDeployment.reasons.length > 0 ? (
                      <p className="mt-2 text-xs text-muted-foreground">
                        {shadowDeployment.reasons.slice(0, 3).map((row: unknown) => String(row)).join(" · ")}
                      </p>
                    ) : null}
                  </div>
                ) : null}

                <div className="max-h-[50vh] overflow-auto rounded-xl border border-border/80">
                  <table className="min-w-full text-xs">
                    <thead className="sticky top-0 z-10 bg-background/95 backdrop-blur">
                      <tr className="text-left text-muted-foreground">
                        <th className="px-3 py-2 font-semibold">Source</th>
                        {([
                          ["symbol", "Symbol"],
                          ["qty", "Qty"],
                          ["entry", "Entry"],
                          ["current", "Current"],
                          ["pnl", "PnL"],
                          ["pnl_pct", "PnL %"],
                          ["signal_direction", "Signal"],
                        ] as [PositionSortKey, string][]).map(([key, label]) => (
                          <th key={key} className="px-3 py-2 font-semibold" aria-sort={sortKey === key ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}>
                            <button
                              type="button"
                              onClick={() => setSort(key)}
                              className="inline-flex items-center gap-1 hover:text-foreground"
                            >
                              {label}
                              <span className="text-[10px]">{sortIndicator(sortKey === key, sortDirection)}</span>
                            </button>
                          </th>
                        ))}
                        <th className="px-3 py-2 font-semibold">Inspect</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedPositions.length === 0 ? (
                        <tr>
                          <td colSpan={9} className="px-3 py-6 text-center text-muted-foreground">
                            {showLoading ? "Loading positions..." : "No positions in API state."}
                          </td>
                        </tr>
                      ) : (
                        sortedPositions.map((position) => (
                          <tr key={positionRowKey(position)} className="border-t border-border/60 hover:bg-secondary/30">
                            <td className="px-3 py-2 text-[11px] uppercase text-muted-foreground">
                              {position.broker_type
                                ? `${String(position.broker_type).toUpperCase()}${position.stale ? " (stale)" : ""}`
                                : (position.source_id ? position.source_id.slice(0, 8) : "state")}
                            </td>
                            <td className="px-3 py-2 font-semibold text-foreground">{position.symbol}</td>
                            <td className="apex-kpi-value px-3 py-2 text-foreground">{position.qty}</td>
                            <td className="apex-kpi-value px-3 py-2 text-foreground">{formatCurrencyWithCents(position.entry)}</td>
                            <td className="apex-kpi-value px-3 py-2 text-foreground">{formatCurrencyWithCents(position.current)}</td>
                            <td className={`apex-kpi-value px-3 py-2 font-semibold ${position.pnl >= 0 ? "text-positive" : "text-negative"}`}>
                              {formatCurrencyWithCents(position.pnl)}
                            </td>
                            <td className={`apex-kpi-value px-3 py-2 font-semibold ${position.pnl_pct >= 0 ? "text-positive" : "text-negative"}`}>
                              {position.pnl_pct.toFixed(2)}%
                            </td>
                            <td className="px-3 py-2 uppercase text-muted-foreground">{position.signal_direction || "unknown"}</td>
                            <td className="px-3 py-2">
                              <Button
                                type="button"
                                variant={replayTarget?.kind === "symbol" && replayTarget?.symbol === position.symbol ? "secondary" : "outline"}
                                size="xs"
                                onClick={() =>
                                  setReplayTarget({
                                    kind: "symbol",
                                    label: position.symbol,
                                    requestUrl: position.replay_url ?? `/api/v1/replay-inspector?symbol=${encodeURIComponent(position.symbol)}`,
                                    symbol: position.symbol,
                                  })
                                }
                              >
                                Inspect
                              </Button>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>

                <ReplayInspectorPanel
                  symbol={replayTarget?.symbol ?? null}
                  planId={replayTarget?.planId ?? null}
                  targetLabel={replayTarget?.label ?? null}
                  loading={replayLoading}
                  error={replayError}
                  data={replayData}
                  onOpenPlan={(planId) =>
                    setReplayTarget({
                      kind: "plan",
                      label: `Plan ${planId}`,
                      requestUrl: `/api/v1/replay-inspector?plan_id=${encodeURIComponent(planId)}`,
                      planId,
                    })
                  }
                  onOpenSymbol={(symbol) =>
                    setReplayTarget({
                      kind: "symbol",
                      label: symbol,
                      requestUrl: `/api/v1/replay-inspector?symbol=${encodeURIComponent(symbol)}`,
                      symbol,
                    })
                  }
                  onClose={() => setReplayTarget(null)}
                />

                <div className="mt-4">
                  <div className="mb-2 flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-foreground">Derivatives (Options)</h3>
                    <span className="text-xs text-muted-foreground">{derivatives.length} option legs</span>
                  </div>
                  <div className="max-h-[30vh] overflow-auto rounded-xl border border-border/80">
                    <table className="min-w-full text-xs">
                      <thead className="sticky top-0 z-10 bg-background/95 backdrop-blur">
                        <tr className="text-left text-muted-foreground">
                          <th className="px-3 py-2 font-semibold">Symbol</th>
                          <th className="px-3 py-2 font-semibold">Expiry</th>
                          <th className="px-3 py-2 font-semibold">Type</th>
                          <th className="px-3 py-2 font-semibold">Strike</th>
                          <th className="px-3 py-2 font-semibold">Qty</th>
                          <th className="px-3 py-2 font-semibold">Avg Cost</th>
                          <th className="px-3 py-2 font-semibold">Side</th>
                        </tr>
                      </thead>
                      <tbody>
                        {derivatives.length === 0 ? (
                          <tr>
                            <td colSpan={7} className="px-3 py-6 text-center text-muted-foreground">
                              {showLoading ? "Loading option legs..." : "No option legs in exported state."}
                            </td>
                          </tr>
                        ) : (
                          derivatives.map((leg: CockpitDerivative) => (
                            <tr key={`${leg.symbol}-${leg.expiry}-${leg.strike}-${leg.right}`} className="border-t border-border/60 hover:bg-secondary/30">
                              <td className="px-3 py-2 font-semibold text-foreground">{leg.symbol}</td>
                              <td className="px-3 py-2 text-foreground">{leg.expiry}</td>
                              <td className="px-3 py-2 text-foreground">{leg.right === "C" ? "CALL" : leg.right === "P" ? "PUT" : leg.right}</td>
                              <td className="apex-kpi-value px-3 py-2 text-foreground">{leg.strike.toFixed(2)}</td>
                              <td className="apex-kpi-value px-3 py-2 text-foreground">{leg.quantity}</td>
                              <td className="apex-kpi-value px-3 py-2 text-foreground">{formatCurrencyWithCents(leg.avg_cost)}</td>
                              <td className="px-3 py-2 text-muted-foreground">{leg.side}</td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </PositionsTable>
            </section>

            {/* ── Signal Heatmap ── */}
            <SignalHeatmap positions={(wsData?.positions ?? null) as Record<string, unknown> | null} />
          </>
        ) : null}
      </div>
    </main>
  );
}
