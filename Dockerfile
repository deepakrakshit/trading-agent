# CPU fallback image. The host's NVIDIA kernel driver is not modified here.
FROM python:3.14.4-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates make libgomp1 libstdc++6 tzdata \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock pyproject.toml README.md ./
RUN python -m venv .venv && .venv/bin/python -m pip install --no-cache-dir -r requirements.lock
COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY Makefile ./
RUN .venv/bin/python -m pip install --no-deps --no-build-isolation -e . \
    && make test \
    && mkdir -p data/raw data/processed models runs \
    && groupadd --gid 10001 trader \
    && useradd --uid 10001 --gid trader --no-create-home trader \
    && chown -R trader:trader /app
USER trader
CMD [".venv/bin/python", "-m", "trading_agent.cli", "demo"]
