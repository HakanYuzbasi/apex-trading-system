import os
import pytest
from importlib import reload
from unittest.mock import patch

# Mock load_dotenv to prevent it from messing with our tests when we reload config
with patch("dotenv.load_dotenv", return_value=None):
    import config

def test_apex_environment_key_reads_correctly(monkeypatch):
    """Test that APEX_ENVIRONMENT takes precedence."""
    monkeypatch.setenv("APEX_ENVIRONMENT", "staging")
    monkeypatch.setenv("ENV", "prod")
    # _resolve_environment_name is a function, no need to reload module for it
    assert config._resolve_environment_name() == "staging"

def test_ibkr_host_fallback_works_without_apex_prefix(monkeypatch):
    """Test that IBKR_HOST is used if APEX_IBKR_HOST is missing."""
    monkeypatch.delenv("APEX_IBKR_HOST", raising=False)
    monkeypatch.setenv("IBKR_HOST", "10.0.0.5")
    # Reload to re-evaluate ApexConfig class attributes
    with patch("dotenv.load_dotenv", return_value=None):
        reload(config)
    assert config.ApexConfig.IBKR_HOST == "10.0.0.5"

def test_ibkr_port_fallback_works_without_apex_prefix(monkeypatch):
    """Test that IBKR_PORT is used if APEX_IBKR_PORT is missing."""
    monkeypatch.delenv("APEX_IBKR_PORT", raising=False)
    monkeypatch.setenv("IBKR_PORT", "8888")
    # Reload to re-evaluate ApexConfig class attributes
    with patch("dotenv.load_dotenv", return_value=None):
        reload(config)
    assert config.ApexConfig.IBKR_PORT == 8888
