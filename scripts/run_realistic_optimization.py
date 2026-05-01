"""Run realistic portfolio WFO sweeps for ROI/Sharpe improvements.

The sweep reuses the R17/R18 feature builder but evaluates folds through
``RealisticPortfolioBacktester`` so results include cash, concurrent positions,
daily marked equity, costs, fee-aware gating, HRP sizing, vol circuit behavior,
and exit rules.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yfinance as yf

    _YF_CACHE = ROOT / "data" / ".yfinance_cache"
    _YF_CACHE.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(_YF_CACHE))
except Exception:
    pass

from backtesting.realistic_portfolio_backtester import (
    ExitConfig,
    PortfolioBacktestConfig,
    RealisticPortfolioBacktester,
    VolCircuitConfig,
)
from backtesting.signal_quality_lab import SignalQualityConfig, SignalQualityLab
from r17_train import UNIVERSE, build_features, load_sym


def _train_fold_model(
    price_data: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
    train_dates: pd.DatetimeIndex,
    spy_ret: pd.Series,
) -> tuple[GradientBoostingClassifier, StandardScaler]:
    x_parts = []
    y_parts = []
    date_set = set(train_dates)
    for symbol in symbols:
        df = price_data[symbol].loc[price_data[symbol].index.isin(date_set)]
        if len(df) < 80:
            continue
        x = build_features(
            df,
            macro=False,
            spy_ret=spy_ret.reindex(df.index, fill_value=0.0),
        ).replace([np.inf, -np.inf], np.nan).dropna()
        y = (df["Close"].pct_change(1).shift(-1) > 0).astype(int).reindex(x.index).dropna()
        x = x.loc[y.index]
        if len(x) < 30:
            continue
        x_parts.append(x.reset_index(drop=True))
        y_parts.append(y.reset_index(drop=True))
    if not x_parts:
        raise RuntimeError("No fold training samples were available")

    x_train = pd.concat(x_parts, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    y_train = pd.concat(y_parts, ignore_index=True).reindex(x_train.index).dropna()
    x_train = x_train.loc[y_train.index]

    scaler = StandardScaler()
    xs = scaler.fit_transform(x_train)
    model = GradientBoostingClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(xs, y_train)
    return model, scaler


def _score_oos(
    model: GradientBoostingClassifier,
    scaler: StandardScaler,
    price_data: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
    oos_dates: pd.DatetimeIndex,
    spy_ret: pd.Series,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    fold_prices: Dict[str, pd.DataFrame] = {}
    probabilities: Dict[str, pd.Series] = {}
    date_set = set(oos_dates)
    for symbol in symbols:
        df = price_data[symbol].loc[price_data[symbol].index.isin(date_set)]
        if len(df) < 20:
            continue
        x = build_features(
            df,
            macro=False,
            spy_ret=spy_ret.reindex(df.index, fill_value=0.0),
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            continue
        prob = model.predict_proba(scaler.transform(x))[:, 1]
        fold_prices[symbol] = df
        probabilities[symbol] = pd.Series(prob, index=x.index, dtype=float)
    return fold_prices, probabilities


def _build_oos_features(
    price_data: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
    oos_dates: pd.DatetimeIndex,
    spy_ret: pd.Series,
) -> Dict[str, pd.DataFrame]:
    features = {}
    date_set = set(oos_dates)
    for symbol in symbols:
        df = price_data[symbol].loc[price_data[symbol].index.isin(date_set)]
        if len(df) < 20:
            continue
        x = build_features(
            df,
            macro=False,
            spy_ret=spy_ret.reindex(df.index, fill_value=0.0),
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if not x.empty:
            features[symbol] = x
    return features


def _fold_metrics(result) -> dict:
    return {
        "return_pct": round(result.total_return_pct * 100.0, 3),
        "sharpe": round(result.sharpe, 3),
        "max_drawdown_pct": round(result.max_drawdown_pct * 100.0, 3),
        "trades": result.trades,
        "win_rate": round(result.win_rate, 3),
        "profit_factor": round(result.profit_factor, 3)
        if np.isfinite(result.profit_factor)
        else "inf",
        "exposure_mean": round(result.exposure_mean, 3),
        "blocks": result.diagnostics.get("blocks", {}),
    }


def run_walk_forward(
    price_data: Dict[str, pd.DataFrame],
    config: PortfolioBacktestConfig,
    train_bars: int,
    oos_bars: int,
    include_signal_quality: bool = False,
) -> dict:
    symbols = [s for s in UNIVERSE if s in price_data]
    biz_days = price_data["SPY"].index
    spy_ret = price_data["SPY"]["Close"].pct_change(1)
    fold_results: List[dict] = []

    i = train_bars
    while i + oos_bars <= len(biz_days):
        train_dates = biz_days[max(0, i - train_bars):i]
        oos_dates = biz_days[i:i + oos_bars]
        model, scaler = _train_fold_model(price_data, symbols, train_dates, spy_ret)
        fold_prices, probs = _score_oos(model, scaler, price_data, symbols, oos_dates, spy_ret)
        result = RealisticPortfolioBacktester(fold_prices, probs, config).run()
        metrics = _fold_metrics(result)
        if include_signal_quality:
            fold_features = _build_oos_features(price_data, symbols, oos_dates, spy_ret)
            signal_report = SignalQualityLab(
                fold_prices,
                probs,
                features=fold_features,
                config=SignalQualityConfig(
                    round_trip_cost_bps=(2.0 * config.one_way_cost_bps)
                    + (2.0 * config.half_spread_bps),
                    cost_edge_ratio=config.fee_min_edge_cost_ratio,
                ),
            ).analyze()
            metrics["signal_quality"] = signal_report.to_dict()
        metrics["start"] = str(oos_dates[0].date())
        metrics["end"] = str(oos_dates[-1].date())
        fold_results.append(metrics)
        i += oos_bars

    if not fold_results:
        return {
            "mean_sharpe": 0.0,
            "median_sharpe": 0.0,
            "total_return_pct": 0.0,
            "worst_drawdown_pct": 0.0,
            "positive_folds": 0,
            "n_folds": 0,
            "trades": 0,
            "folds": [],
        }

    sharpes = [float(f["sharpe"]) for f in fold_results]
    returns = [float(f["return_pct"]) for f in fold_results]
    drawdowns = [float(f["max_drawdown_pct"]) for f in fold_results]
    return {
        "mean_sharpe": round(float(np.mean(sharpes)), 3),
        "median_sharpe": round(float(np.median(sharpes)), 3),
        "total_return_pct": round(float(np.sum(returns)), 3),
        "worst_drawdown_pct": round(float(np.min(drawdowns)), 3),
        "positive_folds": int(sum(s > 0 for s in sharpes)),
        "n_folds": len(fold_results),
        "trades": int(sum(int(f["trades"]) for f in fold_results)),
        "folds": fold_results,
    }


def objective_score(summary: dict) -> float:
    """Constrained score: favor Sharpe, penalize fragile drawdowns and thin samples."""
    mean_sharpe = float(summary.get("mean_sharpe", 0.0))
    median_sharpe = float(summary.get("median_sharpe", 0.0))
    worst_dd = float(summary.get("worst_drawdown_pct", 0.0))
    total_return = float(summary.get("total_return_pct", 0.0))
    trades = int(summary.get("trades", 0))
    n_folds = max(1, int(summary.get("n_folds", 1)))
    positive_folds = int(summary.get("positive_folds", 0))

    score = 0.55 * mean_sharpe + 0.35 * median_sharpe + 0.01 * total_return
    score += 0.10 * (positive_folds / n_folds)
    if worst_dd < -15.0:
        score -= abs(worst_dd + 15.0) * 0.10
    if trades < max(20, n_folds * 5):
        score -= 1.0
    return round(float(score), 6)


def candidate_configs(base: PortfolioBacktestConfig, quick: bool = False) -> List[dict]:
    vol_grid = [
        VolCircuitConfig(True, 1.8, 2.7, 3.6, 0.20),
        VolCircuitConfig(True, 2.0, 3.0, 4.0, 0.25),
        VolCircuitConfig(True, 2.2, 3.2, 4.2, 0.35),
    ]
    exit_grid = [
        ExitConfig(0.025, 0.050, 0.025, 0.50, 15),
        ExitConfig(0.030, 0.060, 0.030, 0.50, 20),
        ExitConfig(0.040, 0.080, 0.035, 0.48, 30),
    ]
    fee_grid = [1.25, 1.50, 2.00]
    kelly_grid = [0.35, 0.50, 0.70]
    max_weight_grid = [0.10, 0.15, 0.20]
    hrp_blend_grid = [0.25, 0.40, 0.60]
    threshold_grid = [0.54, 0.55, 0.58]
    if quick:
        vol_grid = vol_grid[:2]
        exit_grid = exit_grid[:2]
        fee_grid = fee_grid[:1]
        kelly_grid = kelly_grid[1:2]
        max_weight_grid = max_weight_grid[1:2]
        hrp_blend_grid = hrp_blend_grid[1:2]
        threshold_grid = threshold_grid[1:2]

    configs = []
    for vol in vol_grid:
        for exits in exit_grid:
            for fee_ratio in fee_grid:
                for kelly in kelly_grid:
                    for max_weight in max_weight_grid:
                        for hrp_blend in hrp_blend_grid:
                            for threshold in threshold_grid:
                                cfg = replace(
                                    base,
                                    signal_threshold=threshold,
                                    kelly_fraction=kelly,
                                    max_position_pct=max_weight,
                                    hrp_signal_blend=hrp_blend,
                                    vol_circuit=vol,
                                    exits=exits,
                                    fee_min_edge_cost_ratio=fee_ratio,
                                )
                                configs.append(
                                    {
                                        "name": (
                                            f"thr_{threshold:g}_kelly_{kelly:g}"
                                            f"_maxw_{max_weight:g}_hrp_{hrp_blend:g}"
                                            f"_vol_{vol.ratio_scale:g}_{vol.ratio_halt:g}_{vol.ratio_close:g}"
                                            f"_x{vol.scale_multiplier:g}"
                                            f"_exit_{exits.stop_loss_pct:g}_{exits.take_profit_pct:g}"
                                            f"_{exits.trailing_pct:g}_{exits.max_hold_bars}"
                                            f"_fee_{fee_ratio:g}"
                                        ),
                                        "config": cfg,
                                    }
                                )
    return configs


def load_price_data(start: str, end: str) -> Dict[str, pd.DataFrame]:
    data = {}
    for symbol in UNIVERSE:
        data[symbol] = load_sym(symbol, start, end)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--train-bars", type=int, default=504)
    parser.add_argument("--oos-bars", type=int, default=126)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--signal-quality", action="store_true")
    parser.add_argument("--out", default="r18_artifacts/realistic_optimization.json")
    args = parser.parse_args()

    price_data = load_price_data(args.start, args.end)
    base = PortfolioBacktestConfig()
    runs = []
    for candidate in candidate_configs(base, quick=args.quick):
        summary = run_walk_forward(
            price_data=price_data,
            config=candidate["config"],
            train_bars=args.train_bars,
            oos_bars=args.oos_bars,
            include_signal_quality=args.signal_quality,
        )
        row = {
            "name": candidate["name"],
            "summary": summary,
            "config": asdict(candidate["config"]),
        }
        row["objective_score"] = objective_score(summary)
        runs.append(row)
        print(
            f"{row['name']}: meanSharpe={summary['mean_sharpe']} "
            f"worstDD={summary['worst_drawdown_pct']}% "
            f"ret={summary['total_return_pct']}% trades={summary['trades']} "
            f"score={row['objective_score']}"
        )

    runs.sort(
        key=lambda r: (
            r["objective_score"],
            r["summary"]["mean_sharpe"],
            r["summary"]["worst_drawdown_pct"],
            r["summary"]["total_return_pct"],
        ),
        reverse=True,
    )
    payload = {"runs": runs, "best": runs[0] if runs else None}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if runs:
        best = runs[0]
        print(
            "BEST "
            f"{best['name']}: meanSharpe={best['summary']['mean_sharpe']} "
            f"worstDD={best['summary']['worst_drawdown_pct']}% "
            f"ret={best['summary']['total_return_pct']}% "
            f"score={best['objective_score']}"
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
