"""Signal-quality diagnostics aligned with the realistic portfolio backtester.

The lab consumes symbol-level probability series and price data, then measures
whether the signal is calibrated, predictive, cost-covering, persistent across
horizons, and useful as a cross-sectional ranker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalQualityConfig:
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    n_bins: int = 5
    top_n: int = 3
    round_trip_cost_bps: float = 18.0
    cost_edge_ratio: float = 1.50


@dataclass
class SignalQualityReport:
    total_observations: int
    calibration: List[dict] = field(default_factory=list)
    decay: Dict[str, dict] = field(default_factory=dict)
    top_n: Dict[str, dict] = field(default_factory=dict)
    cost_labels: Dict[str, dict] = field(default_factory=dict)
    feature_ic: Dict[str, float] = field(default_factory=dict)
    meta_label_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_observations": self.total_observations,
            "calibration": self.calibration,
            "decay": self.decay,
            "top_n": self.top_n,
            "cost_labels": self.cost_labels,
            "feature_ic": self.feature_ic,
            "meta_label_columns": self.meta_label_columns,
        }


class SignalQualityLab:
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        probabilities: Dict[str, pd.Series],
        features: Optional[Dict[str, pd.DataFrame]] = None,
        config: Optional[SignalQualityConfig] = None,
    ) -> None:
        self.price_data = {s: df.sort_index() for s, df in price_data.items()}
        self.probabilities = {s: p.sort_index() for s, p in probabilities.items()}
        self.features = {s: f.sort_index() for s, f in (features or {}).items()}
        self.config = config or SignalQualityConfig()

    def build_meta_label_dataset(self) -> pd.DataFrame:
        rows = []
        max_horizon = max(self.config.horizons)
        required_edge = self._required_edge()

        for symbol, prob_series in self.probabilities.items():
            df = self.price_data.get(symbol)
            if df is None or df.empty or "Close" not in df:
                continue
            close = df["Close"].sort_index()
            feature_frame = self.features.get(symbol)

            for date, prob_raw in prob_series.items():
                if date not in close.index:
                    continue
                loc = close.index.get_loc(date)
                if isinstance(loc, slice) or loc + max_horizon >= len(close):
                    continue
                prob = float(prob_raw)
                if not np.isfinite(prob):
                    continue
                entry = float(close.iloc[loc])
                if entry <= 0:
                    continue
                signed_signal = (prob - 0.5) * 2.0
                direction = 1.0 if signed_signal >= 0 else -1.0
                confidence = max(prob, 1.0 - prob)
                row = {
                    "symbol": symbol,
                    "date": pd.Timestamp(date),
                    "probability": prob,
                    "signed_signal": signed_signal,
                    "confidence": confidence,
                    "direction": int(direction),
                    "entry_price": entry,
                }
                for horizon in self.config.horizons:
                    future = float(close.iloc[loc + horizon])
                    raw_return = future / entry - 1.0
                    directional_return = raw_return * direction
                    row[f"raw_return_{horizon}d"] = raw_return
                    row[f"directional_return_{horizon}d"] = directional_return
                    row[f"direction_correct_{horizon}d"] = directional_return > 0.0
                    row[f"cost_label_{horizon}d"] = directional_return > required_edge
                if feature_frame is not None and date in feature_frame.index:
                    feat_row = feature_frame.loc[date]
                    for col, value in feat_row.items():
                        if np.isscalar(value) and np.isfinite(float(value)):
                            row[f"feature_{col}"] = float(value)
                rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)

    def analyze(self) -> SignalQualityReport:
        dataset = self.build_meta_label_dataset()
        if dataset.empty:
            return SignalQualityReport(total_observations=0)

        calibration = self._calibration(dataset)
        decay = self._decay(dataset)
        top_n = self._top_n(dataset)
        cost_labels = self._cost_labels(dataset)
        feature_ic = self._feature_ic(dataset)
        return SignalQualityReport(
            total_observations=len(dataset),
            calibration=calibration,
            decay=decay,
            top_n=top_n,
            cost_labels=cost_labels,
            feature_ic=feature_ic,
            meta_label_columns=list(dataset.columns),
        )

    def _calibration(self, dataset: pd.DataFrame) -> List[dict]:
        horizon = self.config.horizons[0]
        frame = dataset[["confidence", f"direction_correct_{horizon}d"]].copy()
        if frame["confidence"].nunique() < 2:
            frame["bin"] = "all"
        else:
            frame["bin"] = pd.qcut(
                frame["confidence"],
                q=min(self.config.n_bins, frame["confidence"].nunique()),
                duplicates="drop",
            )
        out = []
        for _, group in frame.groupby("bin", observed=False):
            mean_conf = float(group["confidence"].mean())
            accuracy = float(group[f"direction_correct_{horizon}d"].mean())
            out.append(
                {
                    "count": int(len(group)),
                    "mean_confidence": round(mean_conf, 4),
                    "direction_accuracy": round(accuracy, 4),
                    "calibration_error": round(abs(mean_conf - accuracy), 4),
                }
            )
        return out

    def _decay(self, dataset: pd.DataFrame) -> Dict[str, dict]:
        out = {}
        for horizon in self.config.horizons:
            raw_col = f"raw_return_{horizon}d"
            directional_col = f"directional_return_{horizon}d"
            out[f"{horizon}d"] = {
                "ic": round(self._spearman(dataset["signed_signal"], dataset[raw_col]), 4),
                "avg_directional_return": round(float(dataset[directional_col].mean()), 6),
                "hit_rate": round(float((dataset[directional_col] > 0).mean()), 4),
            }
        return out

    def _top_n(self, dataset: pd.DataFrame) -> Dict[str, dict]:
        out = {}
        ranked = dataset.sort_values(["date", "probability"], ascending=[True, False])
        for horizon in self.config.horizons:
            raw_col = f"raw_return_{horizon}d"
            top = ranked.groupby("date", observed=False).head(self.config.top_n)
            all_avg = float(dataset[raw_col].mean())
            top_avg = float(top[raw_col].mean()) if not top.empty else 0.0
            out[f"{horizon}d"] = {
                "top_n": int(self.config.top_n),
                "observations": int(len(top)),
                "top_n_avg_return": round(top_avg, 6),
                "all_avg_return": round(all_avg, 6),
                "spread": round(top_avg - all_avg, 6),
                "top_n_hit_rate": round(float((top[raw_col] > 0).mean()), 4)
                if not top.empty
                else 0.0,
            }
        return out

    def _cost_labels(self, dataset: pd.DataFrame) -> Dict[str, dict]:
        out = {}
        for horizon in self.config.horizons:
            label = dataset[f"cost_label_{horizon}d"]
            ret = dataset[f"directional_return_{horizon}d"]
            out[f"{horizon}d"] = {
                "positive_rate": round(float(label.mean()), 4),
                "avg_return_when_positive": round(float(ret[label].mean()), 6)
                if bool(label.any())
                else 0.0,
                "required_edge": round(self._required_edge(), 6),
            }
        return out

    def _feature_ic(self, dataset: pd.DataFrame) -> Dict[str, float]:
        horizon = min(5, max(self.config.horizons))
        target = f"directional_return_{horizon}d"
        if target not in dataset:
            target = f"directional_return_{self.config.horizons[0]}d"
        out = {
            "probability": self._spearman(dataset["probability"], dataset[target]),
            "confidence": self._spearman(dataset["confidence"], dataset[target]),
            "signed_signal": self._spearman(dataset["signed_signal"], dataset[target]),
        }
        for col in [c for c in dataset.columns if c.startswith("feature_")]:
            out[col.removeprefix("feature_")] = self._spearman(dataset[col], dataset[target])
        return {k: round(float(v), 4) for k, v in sorted(out.items())}

    def _required_edge(self) -> float:
        return (self.config.round_trip_cost_bps / 10_000.0) * self.config.cost_edge_ratio

    @staticmethod
    def _spearman(x: pd.Series, y: pd.Series) -> float:
        frame = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 3:
            return 0.0
        if frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
            return 0.0
        corr = frame.iloc[:, 0].rank().corr(frame.iloc[:, 1].rank())
        return float(corr) if np.isfinite(corr) else 0.0
