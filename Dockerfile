FROM continuumio/miniconda3:24.7.1-0

COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the Conda environment
RUN echo "conda activate gfm_bench" >> ~/.bashrc
ENV PATH="$PATH:/opt/conda/envs/gfm_bench/bin"

# Create a non-root user and switch to that user
RUN useradd -m benchuser
USER benchuser

WORKDIR /home/benchuser
# COPY --chown=benchuser main.ipynb .

# Expose ports
EXPOSE 8888
EXPOSE 8787

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0"]
