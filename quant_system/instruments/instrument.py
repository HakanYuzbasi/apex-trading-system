from __future__ import annotations

from dataclasses import dataclass

from core.symbols import AssetClass


@dataclass(frozen=True, slots=True, kw_only=True)
class Instrument:
    instrument_id: str
    asset_class: AssetClass
    shortable: bool = True
    margin_requirement: float | None = None

    def __post_init__(self) -> None:
        if not self.instrument_id.strip():
            raise ValueError("instrument_id must be a non-empty string")
        if self.margin_requirement is not None and not 0.0 < self.margin_requirement <= 1.0:
            raise ValueError("margin_requirement must be in (0.0, 1.0] when provided")
