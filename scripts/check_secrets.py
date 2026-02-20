"""Runtime secret hygiene checks for APEX startup and CI smoke tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SECRET_MARKERS = (
    "SECRET",
    "PASSWORD",
    "TOKEN",
    "API_KEY",
    "ROUTING_KEY",
    "MASTER_KEY",
)

WEAK_VALUES = {
    "",
    "changeme",
    "change_me",
    "password",
    "secret",
    "admin",
    "default",
    "none",
    "null",
}


@dataclass
class SecretSpec:
    """Environment secret definition discovered from .env.example."""

    key: str
    optional: bool


def _runtime_env(explicit_env: Optional[str] = None) -> str:
    """Resolve runtime environment name."""
    return (
        explicit_env
        or os.getenv("APEX_ENV")
        or os.getenv("APEX_ENVIRONMENT")
        or "development"
    ).strip().lower()


def _iter_secret_specs(example_file: Path) -> Iterable[SecretSpec]:
    """Yield secret-like environment keys from .env.example."""
    for raw_line in example_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, trailing = line.partition("=")
        key = key.strip()
        if not key:
            continue
        if not any(marker in key for marker in SECRET_MARKERS):
            continue
        optional = "optional" in trailing.lower()
        yield SecretSpec(key=key, optional=optional)


def _normalize(value: str) -> str:
    """Normalize values for weak secret checks."""
    return value.strip().lower().replace("-", "_")


def _is_weak(value: str) -> bool:
    """Return True if value is considered weak/insecure."""
    normalized = _normalize(value)
    if normalized in WEAK_VALUES:
        return True
    if "change_me" in normalized or "changeme" in normalized:
        return True
    return False


def validate_secrets(runtime_env: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> None:
    """Validate configured secrets and raise RuntimeError on weak values."""
    resolved_env = _runtime_env(runtime_env)
    if resolved_env == "development":
        return

    env_map = env or os.environ
    root = Path(__file__).resolve().parents[1]
    env_example = root / ".env.example"

    violations: List[str] = []
    for spec in _iter_secret_specs(env_example):
        raw_value = (env_map.get(spec.key) or "").strip()
        if not raw_value:
            if spec.optional:
                continue
            violations.append(f"{spec.key}: missing")
            continue
        if _is_weak(raw_value):
            violations.append(f"{spec.key}: weak value detected")

    if violations:
        details = "\n".join(f"- {item}" for item in violations)
        raise RuntimeError(
            "Unsafe secrets detected for non-development runtime:\n"
            f"{details}\n"
            "Set strong values before startup."
        )


def main() -> int:
    """CLI entrypoint."""
    validate_secrets()
    print("Secret checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
