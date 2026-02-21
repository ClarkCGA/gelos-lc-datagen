
FROM python:3.12-slim AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml pixi.lock README.md Makefile LICENSE /app/
COPY src/ /app/src/
RUN pixi install

FROM base AS test

COPY tests/ /app/tests/

CMD ["pixi", "run", "make", "test"]

FROM base AS prod

CMD ["pixi", "run", "make"]

FROM quay.io/jupyter/pytorch-notebook:cuda12-python-3.13 AS dev

USER root

RUN apt-get update \
    && apt-get install -y \
    curl \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local sh
ENV PATH="/app/.pixi/envs/default/bin:${PATH}"

WORKDIR /app

CMD ["pixi", "run", "start-notebook.py"]