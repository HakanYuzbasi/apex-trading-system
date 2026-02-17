"""
data/social

Cross-platform social ingestion contract tooling.
"""

from .contract import build_social_risk_inputs, write_social_risk_inputs
from .validator import SocialInputValidationReport, validate_social_risk_inputs

__all__ = [
    "SocialInputValidationReport",
    "build_social_risk_inputs",
    "validate_social_risk_inputs",
    "write_social_risk_inputs",
]
