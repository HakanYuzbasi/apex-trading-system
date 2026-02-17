from services.common.plan_catalog import get_plan_catalog


def test_plan_catalog_sorted_from_free_to_enterprise():
    plans = get_plan_catalog()
    tiers = [plan.tier.value for plan in plans]
    assert tiers == ["free", "basic", "pro", "enterprise"]


def test_pm_cockpit_is_recommended_and_has_usp():
    plans = get_plan_catalog()
    pm = next(plan for plan in plans if plan.code == "pm_cockpit")
    assert pm.recommended is True
    assert "Adaptive governor" in pm.usp
    assert pm.monthly_usd > 0


def test_free_plan_includes_mandate_copilot_preview():
    plans = get_plan_catalog()
    free_plan = next(plan for plan in plans if plan.code == "apex_scout")
    assert "mandate_copilot_preview" in free_plan.feature_limits
    assert free_plan.feature_limits["mandate_copilot_preview"] > 0
