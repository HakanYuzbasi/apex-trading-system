"""health_check.py — Startup and scheduled health verification.
Writes data/health_status.json. If overall == FAIL, live_trader must halt.
Run standalone: python scripts/health_check.py
"""
import hashlib, json, logging, os, shutil, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import schedule
import yfinance as yf

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STATUS_FILE = DATA_DIR / "health_status.json"
MODEL_PATH  = ROOT / "r18_artifacts" / "model.pkl"
EXEC_LOG    = DATA_DIR / "execution_log.jsonl"
LAST_RUN    = DATA_DIR / "last_successful_run.txt"
DISK_MIN_MB = 500
CHECK_INTERVAL_MIN = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s health_check — %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("health_check")

# Stored checksums (written on first successful load, verified thereafter)
_CKSUM_FILE = DATA_DIR / "model_checksums.json"


# ── Individual checks ─────────────────────────────────────────────────────────
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_model_files() -> dict:
    """Verify model.pkl exists and checksum is stable."""
    label = "model_files"
    if not MODEL_PATH.exists():
        log.critical("Model file missing: %s", MODEL_PATH)
        return {"check": label, "ok": False, "detail": "model.pkl missing — halting"}

    current_md5 = _md5(MODEL_PATH)
    stored: dict = json.loads(_CKSUM_FILE.read_text()) if _CKSUM_FILE.exists() else {}

    if "r18" not in stored:
        stored["r18"] = current_md5
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _CKSUM_FILE.write_text(json.dumps(stored))
        log.info("Model checksum stored: %s", current_md5)
    elif stored["r18"] != current_md5:
        log.critical("Model checksum mismatch! Expected %s got %s",
                     stored["r18"], current_md5)
        return {"check": label, "ok": False,
                "detail": f"checksum mismatch: expected={stored['r18']} got={current_md5}"}

    # Verify loadable
    try:
        import pickle
        with MODEL_PATH.open("rb") as f:
            art = pickle.load(f)
        assert "gbm" in art and "scaler" in art
    except Exception as exc:
        log.critical("Model not loadable: %s", exc)
        return {"check": label, "ok": False, "detail": str(exc)}

    return {"check": label, "ok": True, "detail": f"md5={current_md5[:8]}…"}


def check_alpaca_reachable() -> dict:
    """Try to instantiate Alpaca REST and call get_clock()."""
    label = "alpaca_api"
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret  = os.getenv("ALPACA_SECRET_KEY", "")
    base    = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not api_key or not secret:
        return {"check": label, "ok": False, "detail": "ALPACA_API_KEY/SECRET not set"}
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(api_key, secret, base, api_version="v2")
        clock = api.get_clock()
        return {"check": label, "ok": True,
                "detail": f"is_open={clock.is_open}"}
    except Exception as exc:
        log.error("Alpaca unreachable: %s", exc)
        return {"check": label, "ok": False, "detail": str(exc)}


def check_yfinance_reachable() -> dict:
    """Download 1 day of SPY to verify yfinance connectivity."""
    label = "yfinance"
    try:
        df = yf.download("SPY", period="1d", progress=False, auto_adjust=True)
        if df.empty:
            return {"check": label, "ok": False, "detail": "SPY returned empty"}
        return {"check": label, "ok": True, "detail": f"rows={len(df)}"}
    except Exception as exc:
        log.error("yfinance unreachable: %s", exc)
        return {"check": label, "ok": False, "detail": str(exc)}


def check_disk_space() -> dict:
    """Verify free disk space > DISK_MIN_MB."""
    label = "disk_space"
    usage = shutil.disk_usage(ROOT)
    free_mb = usage.free / 1_048_576
    ok = free_mb > DISK_MIN_MB
    if not ok:
        log.warning("Low disk space: %.0f MB free (minimum %d MB)", free_mb, DISK_MIN_MB)
    return {"check": label, "ok": ok, "detail": f"free={free_mb:.0f}MB"}


def check_polygon_reachable() -> dict:
    """Ping Polygon.io /v1/marketstatus/now — WARN on failure (not FAIL)."""
    label = "polygon_api"
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return {"check": label, "ok": False, "detail": "POLYGON_API_KEY not set (WARN)"}
    try:
        import urllib.request
        url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={api_key}"
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
        market = data.get("market", "unknown")
        return {"check": label, "ok": True, "detail": f"market={market}"}
    except Exception as exc:
        log.warning("Polygon unreachable (WARN only): %s", exc)
        return {"check": label, "ok": False, "detail": f"WARN: {exc}"}


def check_alpaca_data_reachable() -> dict:
    """Verify Alpaca Data API responds — WARN on failure (not FAIL)."""
    label = "alpaca_data_api"
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret  = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not secret:
        return {"check": label, "ok": False, "detail": "credentials not set (WARN)"}
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://data.alpaca.markets/v2/stocks/SPY/bars?timeframe=1Day&limit=1",
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        bars = data.get("bars", [])
        return {"check": label, "ok": True, "detail": f"bars_returned={len(bars)}"}
    except Exception as exc:
        log.warning("Alpaca Data API unreachable (WARN only): %s", exc)
        return {"check": label, "ok": False, "detail": f"WARN: {exc}"}


def check_exec_log_writable() -> dict:
    """Confirm execution log path is writable."""
    label = "exec_log_writable"
    sys.path.insert(0, str(ROOT))
    try:
        from scripts.execution_log import ExecutionLog
        ok = ExecutionLog.is_writable()
    except Exception as exc:
        return {"check": label, "ok": False, "detail": str(exc)}
    return {"check": label, "ok": ok,
            "detail": str(EXEC_LOG) if ok else "not writable"}


def check_last_run_recency() -> dict:
    """Warn if last successful run was more than 26 hours ago."""
    label = "last_run_recency"
    if not LAST_RUN.exists():
        return {"check": label, "ok": True, "detail": "no previous run recorded"}
    try:
        ts = datetime.fromisoformat(LAST_RUN.read_text().strip())
        age = datetime.now(timezone.utc) - ts
        ok  = age < timedelta(hours=26)
        if not ok:
            log.warning("Last successful run was %.1f hours ago.", age.total_seconds() / 3600)
        return {"check": label, "ok": ok,
                "detail": f"age={age.total_seconds()/3600:.1f}h"}
    except Exception as exc:
        return {"check": label, "ok": False, "detail": str(exc)}


# ── Aggregate ─────────────────────────────────────────────────────────────────
CHECKS = [
    check_model_files,
    check_alpaca_reachable,
    check_alpaca_data_reachable,
    check_polygon_reachable,
    check_yfinance_reachable,
    check_disk_space,
    check_exec_log_writable,
    check_last_run_recency,
]


def run_health_check(halt_on_fail: bool = False) -> dict:
    """Run all checks, write health_status.json, return result dict."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for fn in CHECKS:
        try:
            r = fn()
        except Exception as exc:
            r = {"check": fn.__name__, "ok": False, "detail": str(exc)}
        results.append(r)
        level = logging.INFO if r["ok"] else logging.WARNING
        log.log(level, "  %-28s %s  %s",
                r["check"], "OK  " if r["ok"] else "FAIL", r.get("detail", ""))

    # Determine severity: model failure → FAIL; others → WARN if any bad
    model_ok   = next((r["ok"] for r in results if r["check"] == "model_files"), True)
    all_ok     = all(r["ok"] for r in results)
    any_fail   = any(not r["ok"] for r in results)

    if not model_ok:
        overall = "FAIL"
    elif any_fail:
        overall = "WARN"
    else:
        overall = "OK"

    status = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "all_checks":  results,
        "overall":     overall,
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    log.info("Health status: %s → written to %s", overall, STATUS_FILE)

    if overall == "FAIL" and halt_on_fail:
        log.critical("Overall health FAIL — halting process as requested.")
        sys.exit(2)

    return status


def _scheduled_check() -> None:
    run_health_check(halt_on_fail=False)


def main() -> None:
    """Run immediately, then every 60 min via schedule."""
    run_health_check(halt_on_fail=True)
    schedule.every(CHECK_INTERVAL_MIN).minutes.do(_scheduled_check)
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    result = run_health_check(halt_on_fail=False)
    sys.exit(0 if result["overall"] != "FAIL" else 2)

print("FILE COMPLETE: health_check.py")
