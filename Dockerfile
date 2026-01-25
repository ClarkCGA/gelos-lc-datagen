FROM continuumio/miniconda3:24.7.1-0

COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the Conda environment
RUN echo "conda activate gelos-lc-datagen" >> ~/.bashrc
ENV PATH="$PATH:/opt/conda/envs/gelos-lc-datagen/bin"
ENV LOCALTILESERVER_CLIENT_PREFIX='proxy/{port}'

WORKDIR /app

# Expose ports
EXPOSE 8888
EXPOSE 8787

