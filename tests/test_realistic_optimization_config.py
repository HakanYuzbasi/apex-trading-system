from scripts.run_realistic_optimization import (
    candidate_configs,
    objective_score,
)
from backtesting.realistic_portfolio_backtester import PortfolioBacktestConfig


def test_candidate_configs_quick_keeps_cross_product_small():
    configs = candidate_configs(PortfolioBacktestConfig(), quick=True)

    assert len(configs) == 4
    assert all(c["config"].hrp_enabled for c in configs)
    assert {c["config"].kelly_fraction for c in configs} == {0.50}
    assert {c["config"].max_position_pct for c in configs} == {0.15}


def test_objective_score_penalizes_bad_drawdown_and_tiny_samples():
    strong = {
        "mean_sharpe": 1.2,
        "median_sharpe": 1.0,
        "worst_drawdown_pct": -8,
        "total_return_pct": 12,
        "trades": 100,
        "positive_folds": 4,
        "n_folds": 5,
    }
    fragile = {
        **strong,
        "worst_drawdown_pct": -35,
        "trades": 3,
    }

    assert objective_score(strong) > objective_score(fragile)
