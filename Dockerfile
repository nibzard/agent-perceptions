# syntax=docker/dockerfile:1
FROM continuumio/miniconda3:latest

WORKDIR /workspace

COPY environment.yml ./
RUN conda env create -f environment.yml

# Activate environment by default
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate ai_survey" >> ~/.bashrc
ENV PATH /opt/conda/envs/ai_survey/bin:$PATH

# Install Quarto (if not in conda)
RUN wget -qO- https://quarto.org/download/latest/quarto-linux-amd64.deb > quarto.deb \
    && apt-get update && apt-get install -y gdebi-core \
    && gdebi -n quarto.deb \
    && rm quarto.deb

# JupyterLab port
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]