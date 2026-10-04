FROM python:3.11-slim

# curl serves the compose healthcheck. No compiler: every runtime dependency in uv.lock
# ships a Linux wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.11.14

# Create unprivileged user before COPY so --chown flags work without a separate layer.
RUN addgroup --system appgroup && \
    adduser --system --no-create-home --ingroup appgroup appuser

WORKDIR /app

# Compile .pyc at build time: at runtime appuser can't write __pycache__ and
# PYTHONDONTWRITEBYTECODE is set, so otherwise every container start recompiles every import.
ENV UV_COMPILE_BYTECODE=1

# Install dependencies before copying source so this layer is cached unless
# pyproject.toml or uv.lock change (not on every source edit).
COPY --chown=appuser:appgroup pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install the project itself. uv installs it editable, which settings.py
# relies on: PROJECT_ROOT (and the data/ directory under it) resolves to /app.
COPY --chown=appuser:appgroup . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# entrypoint.sh runs as root, fixes volume ownership, then exec's as appuser.
COPY --chown=root:root entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
