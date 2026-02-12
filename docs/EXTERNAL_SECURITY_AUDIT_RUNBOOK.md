# External Security Audit Runbook

This runbook prepares a repeatable package for a third-party security firm.

## Scope

- Backend API (`/api`, `/services`, `/core`)
- Auth, token handling, and websocket authorization
- CI/CD and dependency supply-chain controls
- Frontend auth/session handling
- Model artifact integrity and version governance

## Pre-audit checklist

1. Freeze release branch and capture commit SHA.
2. Export architecture and deployment docs.
3. Export dependency inventories:
   - `pip freeze` for backend env
   - `npm ls --all` for frontend
4. Run internal static checks:
   - `pre-commit run --all-files`
   - `venv/bin/python -m pytest -q --no-cov tests/test_api_auth_health.py tests/test_websocket.py`
5. Run model integrity checks:
   - `venv/bin/python -m scripts.generate_model_manifest`
   - `venv/bin/python -m scripts.verify_model_manifest`

## Evidence bundle

Create and share the generated archive from:

```bash
scripts/build_security_audit_bundle.sh
```

The bundle includes:

- `SECURITY.md`, `ARCHITECTURE.md`, `DEPLOYMENT.md`
- Current pre-commit policy
- Latest model manifest
- Targeted security and websocket test reports
- API surface snapshot from OpenAPI

## Third-party engagement deliverables

- Manual penetration testing report (web + websocket)
- Authentication and authorization review
- Cryptography and key management review
- Dependency/vulnerability review
- Severity-ranked findings with remediation guidance

## Remediation SLAs

- Critical: 24 hours
- High: 3 business days
- Medium: 10 business days
- Low: next release cycle
