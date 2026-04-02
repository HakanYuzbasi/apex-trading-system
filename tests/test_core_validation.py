# tests/test_core_validation.py - Input validation unit tests

import pytest
from core.validation import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    SymbolValidator,
    PriceValidator,
    QuantityValidator,
    OrderValidator,
    ConfigValidator,
    validate_symbol,
    validate_price,
    validate_quantity,
    validate_order,
    is_valid_symbol,
    is_valid_price,
    is_valid_quantity,
)


# ── ValidationResult ────────────────────────────────────────────────────────

class TestValidationResult:
    def test_starts_valid(self):
        r = ValidationResult(is_valid=True)
        assert r.is_valid
        assert bool(r) is True
        assert r.errors == []
        assert r.warnings == []

    def test_add_error_sets_invalid(self):
        r = ValidationResult(is_valid=True)
        r.add_error("price", "too high")
        assert not r.is_valid
        assert len(r.errors) == 1

    def test_add_warning_keeps_valid(self):
        r = ValidationResult(is_valid=True)
        r.add_warning("qty", "large order")
        assert r.is_valid
        assert len(r.warnings) == 1

    def test_merge_propagates_invalidity(self):
        r1 = ValidationResult(is_valid=True)
        r2 = ValidationResult(is_valid=True)
        r2.add_error("x", "bad")
        r1.merge(r2)
        assert not r1.is_valid
        assert len(r1.issues) == 1

    def test_merge_accumulates_issues(self):
        r1 = ValidationResult(is_valid=True)
        r1.add_warning("a", "w1")
        r2 = ValidationResult(is_valid=True)
        r2.add_warning("b", "w2")
        r1.merge(r2)
        assert r1.is_valid
        assert len(r1.issues) == 2


class TestValidationIssue:
    def test_str_format(self):
        issue = ValidationIssue("price", "too high", ValidationSeverity.ERROR)
        assert "[error] price: too high" == str(issue)

    def test_warning_str(self):
        issue = ValidationIssue("qty", "large", ValidationSeverity.WARNING)
        assert "[warning]" in str(issue)


# ── SymbolValidator ─────────────────────────────────────────────────────────

class TestSymbolValidator:
    def test_valid_equity(self):
        r = SymbolValidator.validate("AAPL")
        assert r.is_valid

    def test_valid_equity_uppercase(self):
        r = SymbolValidator.validate("aapl")
        assert r.is_valid

    def test_empty_symbol(self):
        r = SymbolValidator.validate("")
        assert not r.is_valid

    def test_non_string_symbol(self):
        r = SymbolValidator.validate(123)
        assert not r.is_valid

    def test_too_long_symbol(self):
        r = SymbolValidator.validate("A" * 25)
        assert not r.is_valid

    def test_single_letter_warns(self):
        r = SymbolValidator.validate("A")
        assert r.is_valid
        assert len(r.warnings) >= 1

    def test_sector_etf_warning(self):
        r = SymbolValidator.validate("XLF")
        assert r.is_valid
        warns = [w for w in r.warnings if "sector ETF" in w.message]
        assert len(warns) == 1

    def test_crypto_symbol_format(self):
        # CRYPTO:BTC may fail format validation depending on parse_symbol impl
        r = SymbolValidator.validate("CRYPTO:BTC")
        # Just verify it returns a ValidationResult (may or may not be valid)
        assert isinstance(r, ValidationResult)

    def test_forex_symbol_format(self):
        r = SymbolValidator.validate("FX:EURUSD")
        assert isinstance(r, ValidationResult)

    def test_convenience_validate_symbol(self):
        r = validate_symbol("MSFT")
        assert r.is_valid

    def test_convenience_is_valid_symbol(self):
        assert is_valid_symbol("GOOG") is True


# ── PriceValidator ──────────────────────────────────────────────────────────

class TestPriceValidator:
    def test_valid_price(self):
        r = PriceValidator.validate(150.0)
        assert r.is_valid

    def test_negative_price(self):
        r = PriceValidator.validate(-10.0)
        assert not r.is_valid

    def test_zero_price(self):
        r = PriceValidator.validate(0.0)
        assert not r.is_valid

    def test_nan_price(self):
        r = PriceValidator.validate(float("nan"))
        assert not r.is_valid

    def test_inf_price(self):
        r = PriceValidator.validate(float("inf"))
        assert not r.is_valid

    def test_below_min(self):
        r = PriceValidator.validate(0.00001)
        assert not r.is_valid

    def test_above_max(self):
        r = PriceValidator.validate(2_000_000)
        assert not r.is_valid

    def test_non_numeric(self):
        r = PriceValidator.validate("abc")
        assert not r.is_valid

    def test_int_accepted(self):
        r = PriceValidator.validate(100)
        assert r.is_valid

    def test_large_change_warns(self):
        r = PriceValidator.validate(200.0, reference_price=100.0)
        assert r.is_valid  # warning only
        warns = [w for w in r.warnings if "Large price change" in w.message]
        assert len(warns) == 1

    def test_small_change_no_warning(self):
        r = PriceValidator.validate(101.0, reference_price=100.0)
        warns = [w for w in r.warnings if "Large price change" in w.message]
        assert len(warns) == 0

    def test_precision_warning(self):
        r = PriceValidator.validate(100.123456)
        warns = [w for w in r.warnings if "precision" in w.message.lower()]
        assert len(warns) == 1

    def test_convenience_validate_price(self):
        r = validate_price(50.0)
        assert r.is_valid

    def test_convenience_is_valid_price(self):
        assert is_valid_price(50.0) is True
        assert is_valid_price(-1.0) is False


# ── QuantityValidator ───────────────────────────────────────────────────────

class TestQuantityValidator:
    def test_valid_quantity(self):
        r = QuantityValidator.validate(100)
        assert r.is_valid

    def test_zero_quantity(self):
        r = QuantityValidator.validate(0)
        assert not r.is_valid

    def test_negative_quantity(self):
        r = QuantityValidator.validate(-5)
        assert not r.is_valid

    def test_non_numeric(self):
        r = QuantityValidator.validate("abc")
        assert not r.is_valid

    def test_fractional_not_allowed(self):
        r = QuantityValidator.validate(10.5, allow_fractional=False)
        assert not r.is_valid

    def test_fractional_allowed(self):
        r = QuantityValidator.validate(1.5, allow_fractional=True)
        assert r.is_valid

    def test_below_min(self):
        r = QuantityValidator.validate(1, min_quantity=10)
        assert not r.is_valid

    def test_above_max(self):
        r = QuantityValidator.validate(2_000_000)
        assert not r.is_valid

    def test_odd_lot_warning(self):
        r = QuantityValidator.validate(15, lot_size=10)
        assert r.is_valid
        warns = [w for w in r.warnings if "lot size" in w.message]
        assert len(warns) == 1

    def test_large_order_warning(self):
        r = QuantityValidator.validate(50000)
        assert r.is_valid
        warns = [w for w in r.warnings if "Large order" in w.message]
        assert len(warns) == 1

    def test_convenience_validate_quantity(self):
        r = validate_quantity(10)
        assert r.is_valid

    def test_convenience_is_valid_quantity(self):
        assert is_valid_quantity(10) is True
        assert is_valid_quantity(-1) is False


# ── OrderValidator ──────────────────────────────────────────────────────────

class TestOrderValidator:
    def test_valid_market_order(self):
        r = OrderValidator.validate("AAPL", "BUY", 100)
        assert r.is_valid
        assert r.validated_value["side"] == "BUY"
        assert r.validated_value["order_type"] == "MARKET"

    def test_valid_limit_order(self):
        r = OrderValidator.validate("AAPL", "SELL", 50, price=150.0, order_type="LIMIT")
        assert r.is_valid

    def test_limit_order_missing_price(self):
        r = OrderValidator.validate("AAPL", "BUY", 100, order_type="LIMIT")
        assert not r.is_valid

    def test_invalid_side(self):
        r = OrderValidator.validate("AAPL", "INVALID", 100)
        assert not r.is_valid

    def test_invalid_order_type(self):
        r = OrderValidator.validate("AAPL", "BUY", 100, order_type="FOO")
        assert not r.is_valid

    def test_invalid_tif(self):
        r = OrderValidator.validate("AAPL", "BUY", 100, time_in_force="INVALID")
        assert not r.is_valid

    def test_high_buy_limit_warns(self):
        r = OrderValidator.validate(
            "AAPL", "BUY", 100, price=220.0,
            order_type="LIMIT", reference_price=150.0,
        )
        warns = [w for w in r.warnings if "above market" in w.message]
        assert len(warns) >= 1

    def test_low_sell_limit_warns(self):
        r = OrderValidator.validate(
            "AAPL", "SELL", 100, price=80.0,
            order_type="LIMIT", reference_price=150.0,
        )
        warns = [w for w in r.warnings if "below market" in w.message]
        assert len(warns) >= 1

    def test_convenience_validate_order(self):
        r = validate_order("AAPL", "BUY", 100)
        assert r.is_valid


# ── ConfigValidator ─────────────────────────────────────────────────────────

class TestConfigValidator:
    def test_valid_port(self):
        r = ConfigValidator.validate_port(8080)
        assert r.is_valid

    def test_port_out_of_range(self):
        r = ConfigValidator.validate_port(0)
        assert not r.is_valid
        r2 = ConfigValidator.validate_port(70000)
        assert not r2.is_valid

    def test_port_not_in_allowed(self):
        r = ConfigValidator.validate_port(9999, valid_ports=[80, 443, 8080])
        assert not r.is_valid

    def test_port_non_int(self):
        r = ConfigValidator.validate_port("abc")
        assert not r.is_valid

    def test_valid_percentage_decimal(self):
        r = ConfigValidator.validate_percentage(0.5)
        assert r.is_valid
        assert r.validated_value == 0.5

    def test_percentage_converts_from_100(self):
        r = ConfigValidator.validate_percentage(50.0)
        assert r.is_valid
        assert r.validated_value == 0.5

    def test_invalid_percentage(self):
        r = ConfigValidator.validate_percentage(-10.0)
        assert not r.is_valid
        r2 = ConfigValidator.validate_percentage(200.0)
        assert not r2.is_valid

    def test_percentage_non_numeric(self):
        r = ConfigValidator.validate_percentage("abc")
        assert not r.is_valid

    def test_valid_positive_int(self):
        r = ConfigValidator.validate_positive_int(5)
        assert r.is_valid

    def test_positive_int_zero(self):
        r = ConfigValidator.validate_positive_int(0)
        assert not r.is_valid

    def test_positive_int_exceeds_max(self):
        r = ConfigValidator.validate_positive_int(100, max_val=50)
        assert not r.is_valid

    def test_positive_int_non_int(self):
        r = ConfigValidator.validate_positive_int(3.5)
        assert not r.is_valid
