# --- Stage 1: Build Stage ---
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build dependencies for Cython and other C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies and build C extensions
RUN pip install --upgrade pip && \
    pip install --user -r requirements.txt

# Copy source and compile extensions
COPY . .
RUN python3 scripts/compile_modules.py build_ext --inplace

# --- Stage 2: Production Stage ---
FROM python:3.11-slim as runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install runtime dependencies (only libpq for postgres if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source from builder (including compiled extensions)
COPY --from=builder /app /app

# Ensure run_state directory exists for volumes
RUN mkdir -p /app/run_state

EXPOSE 8000 3000 8765

CMD ["sh", "-c", "python3 scripts/live_preflight_check.py && python3 scripts/run_global_harness_v3.py"]
