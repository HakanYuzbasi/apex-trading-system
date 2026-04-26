"""execution_log.py — Append-only JSONL execution log with log rotation.
Each record tracks the full lifecycle: SUBMITTED → FILLED / CANCELLED / FAILED.
"""
import gzip, json, logging, os, shutil
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("execution_log")

DATA_DIR  = Path(__file__).resolve().parents[1] / "data"
LOG_FILE  = DATA_DIR / "execution_log.jsonl"
ROTATE_MB = 10  # archive when > 10 MB


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _maybe_rotate() -> None:
    """Archive execution_log.jsonl when it exceeds ROTATE_MB."""
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > ROTATE_MB * 1_048_576:
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        arch = DATA_DIR / f"execution_log_{ts}.jsonl.gz"
        with LOG_FILE.open("rb") as f_in, gzip.open(arch, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        LOG_FILE.unlink()
        log.info("Execution log rotated → %s", arch)


class ExecutionLog:
    """Write and update execution records in append-only JSONL format."""

    def write(self, symbol: str, direction: str, signal_confidence: float,
              notional: float, status: str, *, regime: str = "",
              vol_cb_state: str = "", kelly_fraction: float = 0.0,
              model_price: float = 0.0, qty: float = 0.0) -> dict:
        """Append a SUBMITTED record; return the record dict."""
        _ensure_dir()
        _maybe_rotate()
        record = {
            "date":              datetime.now(timezone.utc).date().isoformat(),
            "symbol":            symbol,
            "direction":         direction,
            "qty":               round(qty, 6),
            "signal_confidence": round(signal_confidence, 6),
            "model_price":       round(model_price, 4),
            "submitted_at":      datetime.now(timezone.utc).isoformat(),
            "fill_price":        None,
            "fill_time":         None,
            "slippage_bps":      None,
            "regime":            regime,
            "vol_cb_state":      vol_cb_state,
            "kelly_fraction":    round(kelly_fraction, 6),
            "notional":          round(notional, 2),
            "status":            status,
            "_id":               f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}",
        }
        self._append(record)
        log.info("EXEC LOG WRITE  [%s] %s → %s", status, symbol, record["_id"])
        return record

    def update(self, record_id: str, fill_price: float,
               fill_time: str, status: str) -> None:
        """Rewrite the matching record with fill details (O(N) scan, log is small)."""
        _ensure_dir()
        if not LOG_FILE.exists():
            log.error("Cannot update %s — log file missing.", record_id)
            return
        lines = LOG_FILE.read_text().splitlines()
        updated = False
        new_lines = []
        for line in lines:
            rec = json.loads(line)
            if rec.get("_id") == record_id:
                rec["fill_price"] = round(fill_price, 4)
                rec["fill_time"]  = fill_time
                rec["status"]     = status
                if rec.get("model_price") and fill_price:
                    rec["slippage_bps"] = round(
                        abs(fill_price - rec["model_price"]) / rec["model_price"] * 10_000, 2
                    )
                updated = True
            new_lines.append(json.dumps(rec))
        if updated:
            LOG_FILE.write_text("\n".join(new_lines) + "\n")
            log.info("EXEC LOG UPDATE [%s] %s fill=%.4f", status, record_id, fill_price)
        else:
            log.warning("Record %s not found in execution log.", record_id)

    @staticmethod
    def _append(record: dict) -> None:
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def read_all() -> list[dict]:
        """Return all records as a list of dicts."""
        if not LOG_FILE.exists():
            return []
        records = []
        for line in LOG_FILE.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    log.warning("Corrupt JSONL line skipped: %s", exc)
        return records

    @staticmethod
    def is_writable() -> bool:
        """Return True if the log file path is writable (used by health_check)."""
        _ensure_dir()
        try:
            with LOG_FILE.open("a") as _:
                pass
            return True
        except OSError:
            return False


if __name__ == "__main__":
    # Quick self-test
    el = ExecutionLog()
    rec = el.write("SPY", "buy", 0.72, 1500.0, "SUBMITTED",
                   regime="r18", vol_cb_state="normal", kelly_fraction=0.5,
                   model_price=450.0, qty=3.33)
    el.update(rec["_id"], fill_price=450.12, fill_time=datetime.now(timezone.utc).isoformat(),
              status="FILLED")
    print("Records:", len(ExecutionLog.read_all()))

