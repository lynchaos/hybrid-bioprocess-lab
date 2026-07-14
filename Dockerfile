# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

# Install the package with all optional extras except orchestration by default.
# Orchestration (Flyte) is kept separate because it pulls in heavier dependencies.
RUN pip install --no-cache-dir -e ".[tracking,torch,ray,dev]"

# Non-root user for production.
RUN useradd -m -u 1000 hybridbio && chown -R hybridbio:hybridbio /app
USER hybridbio

ENTRYPOINT ["hybridbio"]
CMD ["--help"]
