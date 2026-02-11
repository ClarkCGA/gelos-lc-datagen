
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


FROM quay.io/jupyter/base-notebook:python-3.12 AS dev

ENV NB_UID=1000 \
    NB_GID=100 \
    NB_USER=jovyan \
    CHOWN_HOME=yes \
    CHOWN_HOME_OPTS=-R

USER root

RUN apt-get update \
    && apt-get install -y \
    curl \
    make \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}
RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH="/home/${NB_USER}/.pixi/bin:/app/.pixi/envs/default/bin:${PATH}"

WORKDIR /app

CMD ["pixi", "run", "start-notebook.py"]
