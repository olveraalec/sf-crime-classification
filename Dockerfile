# syntax=docker/dockerfile:1

# Stage 1: Build the isolated Python environment
FROM python:3.11-slim AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

WORKDIR /app

# Copy pinned uv binaries from the official uv image
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

# Copy dependency definitions before source code for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies into /opt/venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
    --locked \
    --no-install-project

# Stage 2: Create the production runtime image
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# XGBoost requires the GNU OpenMP runtime on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a restricted Linux user and writable log directory
RUN groupadd --system appuser \
    && useradd \
        --system \
        --gid appuser \
        --create-home \
        appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app

# Copy only the completed virtual environment from the builder
COPY --from=builder \
    --chown=appuser:appuser \
    /opt/venv \
    /opt/venv

# Copy only files required by the production service
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser config/ config/
COPY --chown=appuser:appuser artifacts/ artifacts/

USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

CMD ["/opt/venv/bin/python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]