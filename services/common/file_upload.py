"""
services/common/file_upload.py - CSV/JSON upload parsing and validation.

Handles file uploads for SaaS features (backtest results, equity curves, etc.).
"""

import csv
import io
import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Maximum upload sizes
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_ROWS = 50_000


class ParsedUpload(BaseModel):
    """Result of parsing an uploaded file."""
    filename: str
    format: str  # "csv" or "json"
    row_count: int
    columns: List[str]
    data: List[Dict[str, Any]]
    warnings: List[str] = Field(default_factory=list)


class EquityCurve(BaseModel):
    """Validated equity curve for backtest analysis."""
    dates: List[str]
    values: List[float]
    returns: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def parse_csv(content: bytes, filename: str = "upload.csv") -> ParsedUpload:
    """Parse CSV bytes into structured data.

    Raises ValueError if the file is malformed or too large.
    """
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    text = content.decode("utf-8-sig")  # Handle BOM
    reader = csv.DictReader(io.StringIO(text))

    if reader.fieldnames is None:
        raise ValueError("CSV has no header row")

    columns = list(reader.fieldnames)
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for i, row in enumerate(reader):
        if i >= MAX_ROWS:
            warnings.append(f"Truncated at {MAX_ROWS} rows")
            break
        # Convert numeric-looking values
        cleaned = {}
        for k, v in row.items():
            if v is None or v == "":
                cleaned[k] = None
            else:
                try:
                    cleaned[k] = float(v)
                    if cleaned[k] == int(cleaned[k]):
                        cleaned[k] = int(cleaned[k])
                except ValueError:
                    cleaned[k] = v
            cleaned[k] = cleaned.get(k, v)
        rows.append(cleaned)

    return ParsedUpload(
        filename=filename,
        format="csv",
        row_count=len(rows),
        columns=columns,
        data=rows,
        warnings=warnings,
    )


def parse_json(content: bytes, filename: str = "upload.json") -> ParsedUpload:
    """Parse JSON bytes into structured data.

    Accepts either a JSON array of objects or a single object with an
    array field (e.g. ``{"results": [...]}``)
    """
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    warnings: List[str] = []

    # Normalize to list of dicts
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        # Look for the first list-valued key
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                rows = val
                warnings.append(f"Extracted array from key '{key}'")
                break
        else:
            # Single object → wrap in list
            rows = [data]
    else:
        raise ValueError("JSON must be an array of objects or an object with array fields")

    if len(rows) > MAX_ROWS:
        rows = rows[:MAX_ROWS]
        warnings.append(f"Truncated at {MAX_ROWS} rows")

    columns = list(rows[0].keys()) if rows else []

    return ParsedUpload(
        filename=filename,
        format="json",
        row_count=len(rows),
        columns=columns,
        data=rows,
        warnings=warnings,
    )


def parse_upload(content: bytes, filename: str) -> ParsedUpload:
    """Auto-detect format and parse uploaded file."""
    lower = filename.lower()
    if lower.endswith(".csv"):
        return parse_csv(content, filename)
    elif lower.endswith(".json"):
        return parse_json(content, filename)
    else:
        # Try JSON first, then CSV
        try:
            return parse_json(content, filename)
        except ValueError:
            return parse_csv(content, filename)


def validate_equity_curve(upload: ParsedUpload) -> EquityCurve:
    """Extract and validate an equity curve from parsed upload data.

    Expects columns like ``date``/``timestamp`` and ``equity``/``value``/``pnl``.
    """
    # Find date column
    date_col = _find_column(upload.columns, ["date", "timestamp", "time", "dt"])
    if date_col is None:
        raise ValueError(
            f"No date column found. Expected one of: date, timestamp, time. "
            f"Got: {upload.columns}"
        )

    # Find value column
    value_col = _find_column(upload.columns, ["equity", "value", "pnl", "cumulative_pnl", "nav", "balance"])
    if value_col is None:
        raise ValueError(
            f"No value column found. Expected one of: equity, value, pnl, nav, balance. "
            f"Got: {upload.columns}"
        )

    dates: List[str] = []
    values: List[float] = []

    for row in upload.data:
        d = row.get(date_col)
        v = row.get(value_col)
        if d is None or v is None:
            continue
        dates.append(str(d))
        try:
            values.append(float(v))
        except (ValueError, TypeError):
            continue

    if len(values) < 2:
        raise ValueError("Equity curve must have at least 2 data points")

    # Calculate returns
    returns = [0.0]
    for i in range(1, len(values)):
        if values[i - 1] != 0:
            returns.append((values[i] - values[i - 1]) / values[i - 1])
        else:
            returns.append(0.0)

    return EquityCurve(
        dates=dates,
        values=values,
        returns=returns,
        metadata={
            "source_file": upload.filename,
            "date_column": date_col,
            "value_column": value_col,
            "total_rows": len(values),
        },
    )


def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first column whose lowercase name matches a candidate."""
    lower_map = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None
