from core.entry_policy import (
    evaluate_generator_demotion,
    get_expectancy_sizing_multiplier,
    select_dominant_generator,
    should_block_no_trade_band,
)


def test_select_dominant_generator_uses_largest_absolute_signal():
    assert (
        select_dominant_generator({"institutional": 0.31, "god_level": -0.48, "advanced": 0.22})
        == "god_level"
    )


def test_evaluate_generator_demotion_scales_negative_but_not_terminal_bucket():
    ledger = {
        "by_bucket": {
            "EQUITY|risk_on|god_level": {
                "trades": 8,
                "avg_pnl_bps": -12.5,
            }
        }
    }
    decision = evaluate_generator_demotion(
        ledger,
        asset_class="EQUITY",
        regime="risk_on",
        generator_signals={"god_level": -0.42, "institutional": -0.21},
        enabled=True,
        min_trades=5,
        block_pnl_bps=-35.0,
        signal_multiplier=0.85,
        confidence_multiplier=0.80,
    )
    assert decision.action == "scale"
    assert decision.dominant_generator == "god_level"
    assert decision.signal_multiplier == 0.85
    assert decision.confidence_multiplier == 0.80


def test_evaluate_generator_demotion_blocks_materially_negative_bucket():
    ledger = {
        "by_bucket": {
            "CRYPTO|high_vol|advanced": {
                "trades": 6,
                "avg_pnl_bps": -48.0,
            }
        }
    }
    decision = evaluate_generator_demotion(
        ledger,
        asset_class="CRYPTO",
        regime="high_vol",
        generator_signals={"advanced": 0.33},
        enabled=True,
        min_trades=5,
        block_pnl_bps=-35.0,
        signal_multiplier=0.85,
        confidence_multiplier=0.80,
    )
    assert decision.action == "block"
    assert decision.trades == 6
    assert decision.avg_pnl_bps == -48.0


def test_no_trade_band_blocks_flat_marginal_signal():
    assert should_block_no_trade_band(
        signal=0.205,
        effective_signal_threshold=0.20,
        slope=0.0,
        band_ratio=0.12,
        max_slope=0.01,
    ) is True


def test_no_trade_band_allows_clear_breakout_signal():
    assert should_block_no_trade_band(
        signal=0.28,
        effective_signal_threshold=0.20,
        slope=0.04,
        band_ratio=0.12,
        max_slope=0.01,
    ) is False


def test_expectancy_sizing_multiplier_penalizes_only_negative_eligible_bucket():
    multiplier = get_expectancy_sizing_multiplier(
        {"trades": 7, "avg_pnl_bps": -25.0},
        min_trades=5,
        loss_floor_bps=-50.0,
        size_floor=0.75,
    )
    assert 0.75 <= multiplier < 1.0


def test_expectancy_sizing_multiplier_ignores_small_sample():
    assert (
        get_expectancy_sizing_multiplier(
            {"trades": 2, "avg_pnl_bps": -60.0},
            min_trades=5,
            loss_floor_bps=-50.0,
            size_floor=0.75,
        )
        == 1.0
    )
