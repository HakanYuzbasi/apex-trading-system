from execution.ibkr_connector import IBKRConnector


def test_ibkr_finite_float_sanitizes_invalid_values():
    connector = IBKRConnector.__new__(IBKRConnector)
    connector.ib = type(
        "_DummyIB",
        (),
        {"isConnected": lambda self: False, "disconnect": lambda self: None},
    )()

    assert connector._finite_float(1.23) == 1.23
    assert connector._finite_float("4.56") == 4.56
    assert connector._finite_float(float("nan"), 0.01) == 0.01
    assert connector._finite_float(float("inf"), 0.02) == 0.02
    assert connector._finite_float("-inf", 0.03) == 0.03
