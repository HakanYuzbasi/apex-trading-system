"""
services/common/plan_catalog.py - Commercial plan catalog for APEX SaaS.

Separates user-facing plan branding from internal subscription tiers.
"""

from __future__ import annotations

from typing import List

from services.common.schemas import PlanOffer, SubscriptionTier
from services.common.subscription import get_features_for_tier


def get_plan_catalog() -> List[PlanOffer]:
    """Return branded subscription plans with tier-mapped capabilities."""
    return [
        PlanOffer(
            code="apex_scout",
            name="APEX Scout",
            tier=SubscriptionTier.FREE,
            monthly_usd=0,
            annual_usd=0,
            recommended=False,
            target_user="Research and paper-trading users",
            usp="Institutional guardrails from day one with zero-cost onboarding.",
            feature_highlights=[
                "PM cockpit dashboard with live risk status",
                "Governor + kill-switch visibility",
                "AI mandate copilot preview (policy recommendations)",
                "Basic account/security management",
            ],
            feature_limits=get_features_for_tier(SubscriptionTier.FREE),
        ),
        PlanOffer(
            code="quant_pro",
            name="Quant Pro",
            tier=SubscriptionTier.BASIC,
            monthly_usd=299,
            annual_usd=2990,
            recommended=False,
            target_user="Independent quants and small systematic desks",
            usp="Cost-aware execution simulation and backtest validation for edge preservation.",
            feature_highlights=[
                "Backtest validator with institutional diagnostics",
                "Execution simulator with friction realism",
                "AI mandate copilot preview with higher query quota",
                "Higher daily API limits for research loops",
            ],
            feature_limits=get_features_for_tier(SubscriptionTier.BASIC),
        ),
        PlanOffer(
            code="pm_cockpit",
            name="PM Cockpit",
            tier=SubscriptionTier.PRO,
            monthly_usd=1499,
            annual_usd=14990,
            recommended=True,
            target_user="Multi-sleeve PM teams targeting >1.5 Sharpe",
            usp="Adaptive governor + attribution loop tuned for hedge-fund-grade live performance.",
            feature_highlights=[
                "Sleeve-level attribution and drift monitoring",
                "Portfolio allocator with cross-sleeve controls",
                "AI mandate copilot with sleeve-aware risk policy output",
                "Workflow pack initiation + PM/compliance sign-off",
                "Advanced model governance workflow support",
            ],
            feature_limits=get_features_for_tier(SubscriptionTier.PRO),
        ),
        PlanOffer(
            code="fund_os",
            name="Fund OS",
            tier=SubscriptionTier.ENTERPRISE,
            monthly_usd=4999,
            annual_usd=49990,
            recommended=False,
            target_user="Institutional funds and regulated asset managers",
            usp="Full-stack operational resilience, compliance tooling, and unlimited scale paths.",
            feature_highlights=[
                "Compliance copilot and audit workflows",
                "AI mandate copilot at institutional throughput",
                "Monthly model-risk reporting with policy-change audit",
                "Unlimited feature throughput and enterprise controls",
                "Multi-team governance and production support posture",
            ],
            feature_limits=get_features_for_tier(SubscriptionTier.ENTERPRISE),
        ),
    ]
