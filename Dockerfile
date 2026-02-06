FROM jupyter/base-notebook:python-3.11

ENV NB_UID=1000 \
    NB_GID=1000 \
    NB_USER=jovyan \
    CHOWN_HOME=yes \
    CHOWN_HOME_OPTS=-R

USER root

RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}
RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH="/home/${NB_USER}/.pixi/bin:${PATH}"

WORKDIR /app

CMD ["start-notebook.sh"]
