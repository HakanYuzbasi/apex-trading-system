# services/common/ - Shared infrastructure for all SaaS features
#
# Modules:
#   db.py           - Async PostgreSQL engine + session management
#   redis_client.py - Redis singleton + rate-limit helpers
#   subscription.py - Subscription tier gating (FastAPI dependencies)
#   schemas.py      - Shared Pydantic models (tiers, jobs, responses)
#   pdf_report.py   - PDF generation via reportlab
#   file_upload.py  - CSV/JSON upload parsing + validation
