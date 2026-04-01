import importlib

import pytest

import config as config_module


def _reload_config(monkeypatch, **env):
    for key in [
        "APEX_ENVIRONMENT",
        "APEX_ENV",
        "ENV",
        "APEX_POLL_INTERVAL_SECONDS",
        "APEX_PUBLIC_WS_POLL_INTERVAL_SECONDS",
        "APEX_RETRAIN_INTERVAL_SECONDS",
        "APEX_HOT_PATH_PROFILING_ENABLED",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    return importlib.reload(config_module)


@pytest.fixture(autouse=True)
def _restore_default_config(monkeypatch):
    yield
    for key in [
        "APEX_ENVIRONMENT",
        "APEX_ENV",
        "ENV",
        "APEX_POLL_INTERVAL_SECONDS",
        "APEX_PUBLIC_WS_POLL_INTERVAL_SECONDS",
        "APEX_RETRAIN_INTERVAL_SECONDS",
        "APEX_HOT_PATH_PROFILING_ENABLED",
    ]:
        monkeypatch.delenv(key, raising=False)
    importlib.reload(config_module)


def test_development_defaults_use_lighter_cadence(monkeypatch):
    module = _reload_config(monkeypatch, ENV="development")

    assert module.ApexConfig.IS_DEVELOPMENT is True
    assert module.ApexConfig.POLL_INTERVAL_SECONDS == 3.0
    assert module.ApexConfig.PUBLIC_WS_POLL_INTERVAL_SECONDS == 5.0
    assert module.ApexConfig.RETRAIN_INTERVAL_SECONDS == 7 * 24 * 60 * 60
    assert module.ApexConfig.HOT_PATH_PROFILING_ENABLED is True


def test_production_defaults_keep_fast_cadence(monkeypatch):
    module = _reload_config(monkeypatch, ENV="production")

    assert module.ApexConfig.IS_DEVELOPMENT is False
    assert module.ApexConfig.POLL_INTERVAL_SECONDS == 1.0
    assert module.ApexConfig.PUBLIC_WS_POLL_INTERVAL_SECONDS == 1.0
    assert module.ApexConfig.RETRAIN_INTERVAL_SECONDS == 24 * 60 * 60
    assert module.ApexConfig.HOT_PATH_PROFILING_ENABLED is False


def test_explicit_overrides_win_over_environment_defaults(monkeypatch):
    module = _reload_config(
        monkeypatch,
        ENV="production",
        APEX_POLL_INTERVAL_SECONDS="2.5",
        APEX_PUBLIC_WS_POLL_INTERVAL_SECONDS="4.0",
        APEX_RETRAIN_INTERVAL_SECONDS="7200",
        APEX_HOT_PATH_PROFILING_ENABLED="true",
    )

    assert module.ApexConfig.POLL_INTERVAL_SECONDS == 2.5
    assert module.ApexConfig.PUBLIC_WS_POLL_INTERVAL_SECONDS == 4.0
    assert module.ApexConfig.RETRAIN_INTERVAL_SECONDS == 7200
    assert module.ApexConfig.HOT_PATH_PROFILING_ENABLED is True
