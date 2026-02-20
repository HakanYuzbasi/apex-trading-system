"""Enforce explicit exception swallow annotations in critical paths."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

CRITICAL_FILES = [
    Path("api/auth.py"),
    Path("services/auth/service.py"),
    Path("services/broker/service.py"),
]

EXCEPT_RE = re.compile(r"^\s*except\s+Exception(?:\s+as\s+\w+)?\s*:\s*(?:#.*)?$")


def _violations_for_file(path: Path) -> List[Tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    violations: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if not EXCEPT_RE.match(line):
            continue
        if "# SWALLOW:" in line:
            continue

        next_non_empty: List[str] = []
        for j in range(idx, len(lines)):
            candidate = lines[j].strip()
            if not candidate or candidate.startswith("#"):
                continue
            next_non_empty.append(candidate)
            if len(next_non_empty) >= 3:
                break

        if any(code_line.startswith("raise") for code_line in next_non_empty):
            continue
        violations.append((idx, line.strip()))
    return violations


def main() -> int:
    all_violations: List[str] = []
    for path in CRITICAL_FILES:
        if not path.exists():
            continue
        for line_number, content in _violations_for_file(path):
            all_violations.append(f"{path}:{line_number}: {content}")

    if all_violations:
        print("Exception policy violations found:")
        for row in all_violations:
            print(row)
        return 1

    print("Exception policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
