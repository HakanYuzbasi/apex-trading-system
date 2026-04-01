"""
tests/test_monitoring_model_registry.py — Unit tests for monitoring/model_registry.py
Champion/challenger version tracking with auto-rollback.
"""
import json
import time
from pathlib import Path

import pytest

from monitoring.model_registry import (
    ModelRegistry,
    ModelVersion,
    PromotionEvent,
    IC_PROMOTE_DELTA,
    IC_ROLLBACK_THRESH,
    SHARPE_MIN,
    _STATUS_CHAMPION,
    _STATUS_CHALLENGER,
    _STATUS_RETIRED,
    _STATUS_ROLLED_BACK,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_registry(tmp_path=None) -> ModelRegistry:
    if tmp_path:
        return ModelRegistry(state_dir=tmp_path)
    return ModelRegistry()


def good_metrics(ic=0.12, sharpe=1.4) -> dict:
    return {"ic": ic, "sharpe": sharpe, "hit_rate": 0.55}


# ─── TestRegisterVersion ──────────────────────────────────────────────────────

class TestRegisterVersion:
    def test_returns_version_id(self):
        reg = make_registry()
        vid = reg.register_version("model_a", good_metrics())
        assert isinstance(vid, str)
        assert "model_a" in vid

    def test_version_stored(self):
        reg = make_registry()
        vid = reg.register_version("model_a", good_metrics())
        assert vid in reg._versions

    def test_status_is_challenger(self):
        reg = make_registry()
        vid = reg.register_version("model_a", good_metrics())
        assert reg._versions[vid].status == _STATUS_CHALLENGER

    def test_metrics_stored(self):
        reg = make_registry()
        vid = reg.register_version("model_a", {"ic": 0.15, "sharpe": 1.5})
        assert reg._versions[vid].metrics["ic"] == 0.15

    def test_event_recorded(self):
        reg = make_registry()
        reg.register_version("model_a", good_metrics())
        assert len(reg._events) == 1
        assert reg._events[0].action == "register"

    def test_unique_ids_per_call(self):
        reg = make_registry()
        vid1 = reg.register_version("model_a", good_metrics())
        time.sleep(0.01)
        vid2 = reg.register_version("model_a", good_metrics())
        assert vid1 != vid2


# ─── TestShouldPromote ────────────────────────────────────────────────────────

class TestShouldPromote:
    def test_first_version_always_promotes(self):
        reg = make_registry()
        vid = reg.register_version("model_a", good_metrics(ic=0.01, sharpe=0.05))
        assert reg.should_promote(vid) is True

    def test_better_ic_promotes(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", good_metrics(ic=0.10 + IC_PROMOTE_DELTA + 0.001))
        assert reg.should_promote(v2) is True

    def test_marginal_ic_does_not_promote(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", good_metrics(ic=0.10 + IC_PROMOTE_DELTA - 0.001))
        assert reg.should_promote(v2) is False

    def test_low_sharpe_does_not_promote(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", {"ic": 0.20, "sharpe": SHARPE_MIN - 0.01})
        assert reg.should_promote(v2) is False

    def test_unknown_version_returns_false(self):
        reg = make_registry()
        assert reg.should_promote("nonexistent") is False


# ─── TestPromote ─────────────────────────────────────────────────────────────

class TestPromote:
    def test_champion_status_set(self):
        reg = make_registry()
        vid = reg.register_version("model_a", good_metrics())
        reg.promote(vid)
        assert reg._versions[vid].status == _STATUS_CHAMPION

    def test_previous_champion_retired(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", good_metrics(ic=0.15))
        reg.promote(v2)
        assert reg._versions[v1].status == _STATUS_RETIRED

    def test_champions_dict_updated(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        assert reg._champions["model_a"] == v1
        v2 = reg.register_version("model_a", good_metrics(ic=0.15))
        reg.promote(v2)
        assert reg._champions["model_a"] == v2

    def test_promote_records_event(self):
        reg = make_registry()
        vid = reg.register_version("model_a", good_metrics())
        reg.promote(vid)
        promote_events = [e for e in reg._events if e.action == "promote"]
        assert len(promote_events) == 1

    def test_promote_unknown_version_no_crash(self):
        reg = make_registry()
        reg.promote("nonexistent")  # Should not raise


# ─── TestRollback ─────────────────────────────────────────────────────────────

class TestRollback:
    def test_rollback_returns_previous_version_id(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", good_metrics(ic=0.15))
        reg.promote(v2)
        rolled = reg.rollback("model_a")
        assert rolled == v1

    def test_rolled_back_version_status(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", good_metrics(ic=0.15))
        reg.promote(v2)
        reg.rollback("model_a")
        assert reg._versions[v2].status == _STATUS_ROLLED_BACK

    def test_previous_version_restored_as_champion(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", good_metrics(ic=0.15))
        reg.promote(v2)
        reg.rollback("model_a")
        assert reg._champions["model_a"] == v1
        assert reg._versions[v1].status == _STATUS_CHAMPION

    def test_rollback_no_prior_versions_returns_none(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics())
        reg.promote(v1)
        result = reg.rollback("model_a")
        assert result is None

    def test_rollback_records_event(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", good_metrics(ic=0.15))
        reg.promote(v2)
        reg.rollback("model_a")
        rollback_events = [e for e in reg._events if e.action == "rollback"]
        assert len(rollback_events) == 1


# ─── TestShouldRollback ───────────────────────────────────────────────────────

class TestShouldRollback:
    def test_degraded_ic_with_prior_version_triggers(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", good_metrics(ic=0.12))
        reg.promote(v2)
        # v1 is retired — so rollback candidate exists
        degraded_ic = 0.12 - abs(IC_ROLLBACK_THRESH) - 0.001
        assert reg.should_rollback("model_a", degraded_ic) is True

    def test_marginal_degradation_no_rollback(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        # Slightly below IC_ROLLBACK_THRESH edge — no retired version anyway
        assert reg.should_rollback("model_a", 0.10 - abs(IC_ROLLBACK_THRESH) + 0.001) is False

    def test_no_prior_version_no_rollback_even_if_degraded(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        # Degraded far, but no retired version
        assert reg.should_rollback("model_a", 0.05) is False


# ─── TestGetChampion ──────────────────────────────────────────────────────────

class TestGetChampion:
    def test_returns_none_when_no_champion(self):
        reg = make_registry()
        assert reg.get_champion("model_a") is None

    def test_returns_champion_after_promote(self):
        reg = make_registry()
        vid = reg.register_version("model_a", good_metrics())
        reg.promote(vid)
        champ = reg.get_champion("model_a")
        assert champ is not None
        assert champ.version_id == vid

    def test_returns_new_champion_after_second_promote(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", good_metrics(ic=0.15))
        reg.promote(v2)
        assert reg.get_champion("model_a").version_id == v2


# ─── TestGetVersions ─────────────────────────────────────────────────────────

class TestGetVersions:
    def test_returns_newest_first(self):
        reg = make_registry()
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        time.sleep(0.01)
        v2 = reg.register_version("model_a", good_metrics(ic=0.12))
        versions = reg.get_versions("model_a")
        assert versions[0].version_id == v2

    def test_only_returns_correct_model(self):
        reg = make_registry()
        reg.register_version("model_a", good_metrics())
        reg.register_version("model_b", good_metrics())
        assert all(v.model_name == "model_a" for v in reg.get_versions("model_a"))


# ─── TestGetSnapshot ─────────────────────────────────────────────────────────

class TestGetSnapshot:
    def test_snapshot_keys_present(self):
        reg = make_registry()
        snap = reg.get_snapshot()
        for key in ("available", "total_models", "total_versions", "models", "recent_events",
                    "ic_promote_delta", "ic_rollback_thresh", "sharpe_min"):
            assert key in snap

    def test_available_true(self):
        reg = make_registry()
        assert reg.get_snapshot()["available"] is True

    def test_model_appears_in_snapshot(self):
        reg = make_registry()
        vid = reg.register_version("model_a", good_metrics())
        reg.promote(vid)
        snap = reg.get_snapshot()
        assert "model_a" in snap["models"]
        assert snap["models"]["model_a"]["champion_id"] == vid

    def test_recent_events_in_snapshot(self):
        reg = make_registry()
        vid = reg.register_version("model_a", good_metrics())
        reg.promote(vid)
        snap = reg.get_snapshot()
        assert len(snap["recent_events"]) >= 1


# ─── TestPersistence ─────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        reg = ModelRegistry(state_dir=tmp_path)
        v1 = reg.register_version("model_a", good_metrics(ic=0.10))
        reg.promote(v1)
        v2 = reg.register_version("model_a", good_metrics(ic=0.15))
        reg.promote(v2)

        reg2 = ModelRegistry(state_dir=tmp_path)
        assert reg2._champions.get("model_a") == v2
        assert reg2._versions[v2].status == _STATUS_CHAMPION
        assert reg2._versions[v1].status == _STATUS_RETIRED

    def test_events_persisted(self, tmp_path):
        reg = ModelRegistry(state_dir=tmp_path)
        vid = reg.register_version("model_a", good_metrics())
        reg.promote(vid)
        reg2 = ModelRegistry(state_dir=tmp_path)
        assert len(reg2._events) >= 2  # register + promote

    def test_state_file_valid_json(self, tmp_path):
        reg = ModelRegistry(state_dir=tmp_path)
        reg.register_version("model_a", good_metrics())
        p = tmp_path / "model_registry.json"
        assert p.exists()
        data = json.loads(p.read_text())
        assert "versions" in data
        assert "champions" in data

    def test_load_missing_file_no_crash(self, tmp_path):
        reg = ModelRegistry(state_dir=tmp_path)
        assert len(reg._versions) == 0

    def test_no_state_dir_no_save(self):
        reg = ModelRegistry()
        vid = reg.register_version("model_a", good_metrics())
        reg.promote(vid)
        assert reg._state_dir is None
