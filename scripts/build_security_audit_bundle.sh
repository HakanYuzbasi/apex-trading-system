#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/audit_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_DIR="${OUT_DIR}/security_audit_bundle_${STAMP}"

mkdir -p "${BUNDLE_DIR}"

cp "${ROOT_DIR}/SECURITY.md" "${BUNDLE_DIR}/" || true
cp "${ROOT_DIR}/ARCHITECTURE.md" "${BUNDLE_DIR}/" || true
cp "${ROOT_DIR}/DEPLOYMENT.md" "${BUNDLE_DIR}/" || true
cp "${ROOT_DIR}/.pre-commit-config.yaml" "${BUNDLE_DIR}/" || true
cp "${ROOT_DIR}/docs/EXTERNAL_SECURITY_AUDIT_RUNBOOK.md" "${BUNDLE_DIR}/" || true

if [[ -f "${ROOT_DIR}/models/model_manifest.json" ]]; then
  cp "${ROOT_DIR}/models/model_manifest.json" "${BUNDLE_DIR}/"
fi

if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
  "${ROOT_DIR}/venv/bin/python" -m pytest -q --no-cov \
    "${ROOT_DIR}/tests/test_api_auth_health.py" \
    "${ROOT_DIR}/tests/test_websocket.py" \
    > "${BUNDLE_DIR}/security_test_report.txt" || true
fi

if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
  "${ROOT_DIR}/venv/bin/python" - <<'PY' > "${BUNDLE_DIR}/openapi_snapshot.json" || true
import json
from api.server import app
print(json.dumps(app.openapi(), indent=2))
PY
fi

(
  cd "${OUT_DIR}"
  tar -czf "security_audit_bundle_${STAMP}.tar.gz" "security_audit_bundle_${STAMP}"
)

echo "Created ${OUT_DIR}/security_audit_bundle_${STAMP}.tar.gz"
