# syntax=docker/dockerfile:1

# Stage 1: Build environment
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN <<HEREDOC
    apt-get update
    apt-get install -y --no-install-recommends \
        curl \
        libssl-dev \
        pkg-config

    rm -rf /var/lib/apt/lists/*
HEREDOC

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup update nightly && rustup default nightly

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libomp-dev ca-certificates libssl-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3.11 is default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install maturin==1.7.0

WORKDIR /
RUN git clone https://github.com/EricLBuehler/mistral.rs.git /mistral.rs
WORKDIR /mistral.rs

# Install Python deps (use prebuilt CUDA wheel for mistralrs)
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir runpod 

RUN MISTRAL_CUDA=1 maturin build --release --features="cuda" -m mistralrs-pyo3/Cargo.toml --compatibility=off -o /mistralrs
RUN pip install /mistralrs/mistralrs-*-cp311-cp311-linux_x86_64.whl

# Copy chat templates (optional if mistralrs-cuda already ships them)
WORKDIR /chat_templates
COPY chat_templates /chat_templates

# Copy Python handler
WORKDIR /
COPY handler.py /handler.py

# HuggingFace cache
ENV HF_HOME=/runpod-volume/hf_cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache/transformers \
    HF_DATASETS_CACHE=/runpod-volume/hf_cache/datasets \
    HF_MODULES_CACHE=/runpod-volume/hf_cache/modules \
    HF_ASSETS_CACHE=/runpod-volume/hf_cache/assets \
    PYTHONUNBUFFERED=1

# Ensure NVIDIA runtime visibility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python", "handler.py"]